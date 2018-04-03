import argparse
import dill
import logging
import random
import os
import math
import numpy as np
from collections import OrderedDict

import torch
from torch import cuda
from torch.autograd import Variable

import data
from meters import AverageMeter
from discriminator import Discriminator
from generator import NMT

def train_d(args, dataset, warm_train=False):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

    use_cuda = (len(args.gpuid) >= 1)

    # check checkpoints saving path
    if not os.path.exists('checkpoints/discriminator'):
        os.makedirs('checkpoints/discriminator')

    checkpoints_path = 'checkpoints/discriminator/'

    logging_meters = OrderedDict()
    logging_meters['train_loss'] = AverageMeter()
    logging_meters['train_acc'] = AverageMeter()
    logging_meters['valid_loss'] = AverageMeter()
    logging_meters['valid_acc']  = AverageMeter()
    logging_meters['update_times'] = AverageMeter()

    # Build model
    discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict,  use_cuda=use_cuda)

    # fix discriminator word embedding (as Wu et al. do)
    for p in discriminator.embed_src_tokens.parameters():
        p.requires_grad = False
    for p in discriminator.embed_trg_tokens.parameters():
        p.requires_grad = False

    # Load generator
    assert os.path.exists('checkpoints/generator/best_gmodel.pt')
    generator = torch.load('checkpoints/generator/best_gmodel.pt')
    generator.decoder.is_testing = True
    generator.eval()

    if use_cuda:
        discriminator.cuda()
        generator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()

    criterion = torch.nn.BCELoss()

    optimizer = eval("torch.optim." + args.d_optimizer)(filter(lambda x: x.requires_grad, discriminator.parameters()), args.d_learning_rate, momentum=args.momentum, nesterov=True)

    best_dev_loss = math.inf
    # main training loop
    for epoch_i in range(1, args.epochs + 1):
        logging.info("At {0}-th epoch.".format(epoch_i))

        # seed = args.seed + epoch_i
        # torch.manual_seed(seed)

        max_positions_train = (args.pad_dim, args.pad_dim)

        # Initialize dataloader, starting at batch_offset
        itr = dataset.train_dataloader(
            'train',
            max_tokens=args.max_tokens,
            max_sentences=args.joint_batch_size,
            max_positions=max_positions_train,
            # seed=seed,
            epoch=epoch_i,
            sample_without_replacement=args.sample_without_replacement,
            sort_by_source_size=(epoch_i <= args.curriculum),
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )
        # set training mode
        discriminator.train()

        # # update learning rate if necessary
        # update_learning_rate(epoch_i, args.lr_shrink, args.lr_shrink_from, optimizer)

        # reset meters
        for key, val in logging_meters.items():
            if val is not None:
                val.reset()
        # training process
        for i, sample in enumerate(itr):
            if use_cuda:
                sample['id'] = sample['id'].cuda()
                sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].cuda()
                sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].cuda()
                sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].cuda()
                sample['target'] = sample['target'].cuda()

            # train D with 50% fake translation and 50% human translation
            bsz = sample['target'].size(0)
            src_sentence = sample['net_input']['src_tokens']
            # train with half human-translation and half machine translation

            true_sentence = sample['target']
            true_labels = Variable(torch.ones(sample['target'].size(0)).float())

            with torch.no_grad():
                generator.decoder.is_testing = True
                _, prediction = generator(sample)
                generator.decoder.is_testing = False
            fake_sentence = prediction
            fake_labels = Variable(torch.zeros(sample['target'].size(0)).float())

            trg_sentence = torch.cat([true_sentence, fake_sentence], dim=0)
            labels = torch.cat([true_labels, fake_labels], dim=0)

            indices = np.random.permutation(2 * bsz)
            trg_sentence = trg_sentence[indices][:bsz]
            labels = labels[indices][:bsz]

            disc_out = discriminator(src_sentence, trg_sentence, dataset.dst_dict.pad())
            if use_cuda:
                labels = labels.cuda()

            loss = criterion(disc_out, labels)
            acc = torch.sum(torch.round(disc_out).squeeze(1) == labels).float() / len(labels)
            logging_meters['train_acc'].update(acc)
            logging_meters['train_loss'].update(loss)
            logging.debug("D training loss {0:.3f}, acc {1:.3f} at batch {2}: ".format(logging_meters['train_loss'].avg, logging_meters['train_acc'].avg, i))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if logging_meters['train_acc'].avg >= 0.6:
                torch.save(discriminator.state_dict(), checkpoints_path + "best_dmodel.pt")
                return

        max_positions_valid = (args.pad_dim, args.pad_dim)
        # validation process
        itr = dataset.eval_dataloader(
            'valid',
            max_tokens=args.max_tokens,
            max_sentences=args.joint_batch_size,
            max_positions=max_positions_valid,
            skip_invalid_size_inputs_valid_test=True,
            descending=True,  # largest batch first to warm the caching allocator
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )
        # set validation mode
        discriminator.eval()

        # reset meters
        for key, val in logging_meters.items():
            if val is not None:
                val.reset()

        for i, sample in enumerate(itr):
            with torch.no_grad():
                if use_cuda:
                    sample['id'] = sample['id'].cuda()
                    sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].cuda()
                    sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].cuda()
                    sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].cuda()
                    sample['target'] = sample['target'].cuda()

                bsz = sample['target'].size(0)
                src_sentence = sample['net_input']['src_tokens']
                # train with half human-translation and half machine translation

                true_sentence = sample['target']
                true_labels = Variable(torch.ones(sample['target'].size(0)).float())

                with torch.no_grad():
                    _, prediction = generator(sample)
                fake_sentence = prediction
                fake_labels = Variable(torch.zeros(sample['target'].size(0)).float())

                trg_sentence = torch.cat([true_sentence, fake_sentence], dim=0)
                labels = torch.cat([true_labels, fake_labels], dim=0)

                indices = np.random.permutation(2 * bsz)
                trg_sentence = trg_sentence[indices][:bsz]
                labels = labels[indices][:bsz]

                if use_cuda:
                    labels = labels.cuda()

                disc_out = discriminator(src_sentence, trg_sentence, dataset.dst_dict.pad())

                loss = criterion(disc_out, labels)
                acc = torch.sum(torch.round(disc_out).squeeze(1) == labels).float() / len(labels)
                logging_meters['valid_acc'].update(acc)
                logging_meters['valid_loss'].update(loss)
                logging.debug("D dev loss {0:.3f}, acc {1:.3f} at batch {2}".format(logging_meters['valid_loss'].avg, logging_meters['valid_acc'].avg, i))

        torch.save(discriminator.state_dict(), checkpoints_path + "ce_{0:.3f}.epoch_{1}.pt".format(logging_meters['valid_loss'].avg, epoch_i))

        if logging_meters['valid_loss'].avg < best_dev_loss:
            best_dev_loss = logging_meters['valid_loss'].avg
            torch.save(discriminator.state_dict(), checkpoints_path + "best_dmodel.pt")

        if logging_meters['valid_acc'].avg >= 0.6:
            break



def update_learning_rate(current_epoch, lr_shrink, lr_shrink_from, optimizer):
    if (current_epoch == lr_shrink_from):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_shrink