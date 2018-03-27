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

def train_d(args, dataset):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

    use_cuda = (len(args.gpuid) >= 1)
    if args.gpuid:
        cuda.set_device(args.gpuid[0])

    # check checkpoints saving path
    if not os.path.exists('checkpoints/discriminator'):
        os.makedirs('checkpoints/discriminator')

    checkpoints_path = 'checkpoints/discriminator/'

    logging_meters = OrderedDict()
    logging_meters['train_loss'] = AverageMeter()
    logging_meters['valid_loss'] = AverageMeter()
    logging_meters['valid_acc']  = AverageMeter
    logging_meters['update_times'] = AverageMeter()

    # Build model
    discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict,  use_cuda=use_cuda)

    # Load generator
    assert os.path.exists('checkpoints/generator/best_g_model.pt')
    generator = torch.load('checkpoints/generator/best_g_model.pt')
    generator.decoder.is_testing = True
    generator.eval()

    if use_cuda:
        discriminator.cuda()
        generator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()

    criterion = torch.nn.BCELoss()

    optimizer = eval("torch.optim." + args.optimizer)(discriminator.parameters(), args.learning_rate)

    best_dev_loss = math.inf
    # main training loop
    for epoch_i in range(1, args.epochs + 1):
        logging.info("At {0}-th epoch.".format(epoch_i))

        seed = args.seed + epoch_i
        torch.manual_seed(seed)

        max_positions_train = (args.max_source_positions, args.max_target_positions)

        # Initialize dataloader, starting at batch_offset
        itr = dataset.train_dataloader(
            'train',
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=max_positions_train,
            seed=seed,
            epoch=epoch_i,
            sample_without_replacement=args.sample_without_replacement,
            sort_by_source_size=(epoch_i <= args.curriculum),
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )
        # set training mode
        discriminator.train()

        # # update learning rate if necessary
        update_learning_rate(epoch_i, args.lr_shrink, args.lr_shrink_from, optimizer)

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


            src_sentence = sample['net_input']['src_tokens']
            # train with human-translation
            if random.random() >= 0.5:
                trg_sentence = sample['net_input']['target']
            # train with fake translation
            else:
                with torch.no_grad():
                    _, prediction = generator(src_sentence)  # (trg_seq_len, batch_size, trg_vocab_size)
                    prediction = prediction.squeeze(0)
                trg_sentence = prediction

            disc_out = discriminator(src_sentence, trg_sentence, dataset.dst_dict.pad())
            labels = Variable(torch.ones(sample['target'].size(0)).long())
            if use_cuda:
                labels = labels.cuda()

            loss = criterion(disc_out, labels)
            logging.debug("Discriminator training loss at batch {0}: {1:.3f}".format(i, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # validation process
        itr = dataset.eval_dataloader(
            'valid',
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=(50, 50),
            skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test,
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

                src_sentence = sample['net_input']['src_tokens']
                # train with human-translation
                rand = random.random()
                if rand >= 0.5:
                    trg_sentence = sample['net_input']['target']
                    labels = Variable(torch.ones(sample['target'].size(0)).long())
                # train with fake translation
                else:
                    _, prediction = generator(src_sentence)  # (trg_seq_len, batch_size, trg_vocab_size)
                    prediction = prediction.squeeze(0)
                    trg_sentence = Variable(prediction.data, requires_grad=True)
                    labels = Variable(torch.zeros(sample['target'].size(0)).long())

                disc_out = discriminator(src_sentence, trg_sentence, dataset.dst_dict.pad())

                if use_cuda:
                    labels = labels.cuda()

                loss = criterion(disc_out, labels)
                top_val, top_inx = disc_out.topk(1)
                acc = np.sum(top_inx == labels) / len(labels)
                logging_meters['valid_acc'].update(acc)
                logging_meters['valid_loss'].update(loss)
                logging.debug("Discriminator dev loss at batch {0}: {1:.3f}".format(i, loss.item()))
                logging.debug("Discriminator dev accuracy at batch {0}: {1:.3f}".format(i, acc.item()))

        torch.save(discriminator.state_dict(), checkpoints_path + "ce_{0:.3f}.epoch_{1}.pt".format(logging_meters['valid_loss'].avg, epoch_i))

        if logging_meters['valid_loss'].avg < best_dev_loss:
            best_dev_loss = logging_meters['valid_loss']
            torch.save(discriminator.state_dict(), checkpoints_path + "best_d_model.pt")

        if logging_meters['valid_acc'].avg >= 0.6:
            break



def update_learning_rate(current_epoch, lr_shrink, lr_shrink_from, optimizer):
    if (current_epoch == lr_shrink_from):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_shrink