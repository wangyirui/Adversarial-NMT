import argparse
import logging
import math
import dill
import os
import options
import random
import numpy as np
from collections import OrderedDict

import torch
from torch import cuda
from torch.autograd import Variable

import data
import utils
from meters import AverageMeter
from discriminator import Discriminator
from generator import LSTMModel
from train_generator import train_g
from train_discriminator import train_d
from PGLoss import PGLoss



logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Driver program for JHU Adversarial-NMT.")

# Load args
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_distributed_training_args(parser)
options.add_optimization_args(parser)
options.add_checkpoint_args(parser)
options.add_generator_model_args(parser)
options.add_discriminator_model_args(parser)
options.add_generation_args(parser)

def main(args):
    use_cuda = (len(args.gpuid) >= 1)

    # Load dataset
    splits = ['train', 'valid']
    if data.has_binary_files(args.data, splits):
        dataset = data.load_dataset(
            args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
    else:
        dataset = data.load_raw_text_dataset(
            args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
    if args.src_lang is None or args.trg_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.src_lang, args.trg_lang = dataset.src, dataset.dst
    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    for split in splits:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    g_logging_meters = OrderedDict()
    g_logging_meters['train_loss'] = AverageMeter()
    g_logging_meters['valid_loss'] = AverageMeter()
    g_logging_meters['train_acc'] = AverageMeter()
    g_logging_meters['valid_acc'] = AverageMeter()
    g_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    d_logging_meters = OrderedDict()
    d_logging_meters['train_loss'] = AverageMeter()
    d_logging_meters['valid_loss'] = AverageMeter()
    d_logging_meters['train_acc'] = AverageMeter()
    d_logging_meters['valid_acc'] = AverageMeter()
    d_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    # Set model parameters
    args.encoder_embed_dim = 1000
    args.encoder_layers = 4
    args.encoder_dropout_out = 0
    args.decoder_embed_dim = 1000
    args.decoder_layers = 4
    args.decoder_out_embed_dim = 1000
    args.decoder_dropout_out = 0
    args.bidirectional = False

    # try to load generator model
    g_model_path = 'checkpoints/generator/best_gmodel.pt'
    if not os.path.exists(g_model_path):
        print("Start training generator!")
        train_g(args, dataset)
    assert os.path.exists(g_model_path)
    generator = LSTMModel(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    model_dict = generator.state_dict()
    pretrained_dict = torch.load(g_model_path)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    generator.load_state_dict(model_dict)

    print("Generator has successfully loaded!")

    # try to load discriminator model
    d_model_path = 'checkpoints/discriminator/best_dmodel.pt'
    if not os.path.exists(d_model_path):
        print("Start training discriminator!")
        train_d(args, dataset)
    assert  os.path.exists(d_model_path)
    discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    model_dict = discriminator.state_dict()
    pretrained_dict = torch.load(d_model_path)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    generator.load_state_dict(model_dict)

    print("Discriminator has successfully loaded!")

    return

    if use_cuda:
        discriminator.cuda()
        generator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()

    # adversarial training checkpoints saving path
    if not os.path.exists('checkpoints/joint'):
        os.makedirs('checkpoints/joint')
    checkpoints_path = 'checkpoints/joint/'

    # define loss function
    g_criterion = torch.nn.NLLLoss(size_average=False, ignore_index=dataset.dst_dict.pad(),reduce=True)
    d_criterion = torch.nn.BCELoss()
    pg_criterion = PGLoss(ignore_index=dataset.dst_dict.pad(), size_average=True,reduce=True)

    # fix discriminator word embedding (as Wu et al. do)
    for p in discriminator.embed_src_tokens.parameters():
        p.requires_grad = False
    for p in discriminator.embed_trg_tokens.parameters():
        p.requires_grad = False

    # define optimizer
    g_optimizer = eval("torch.optim." + args.g_optimizer)(filter(lambda x: x.requires_grad,
                                                                 generator.parameters()),
                                                          args.g_learning_rate)

    d_optimizer = eval("torch.optim." + args.d_optimizer)(filter(lambda x: x.requires_grad,
                                                                 discriminator.parameters()),
                                                          args.d_learning_rate,
                                                          momentum=args.momentum,
                                                          nesterov=True)

    # start joint training
    best_dev_loss = math.inf
    num_update = 0
    # main training loop
    for epoch_i in range(1, args.epochs + 1):
        logging.info("At {0}-th epoch.".format(epoch_i))

        # seed = args.seed + epoch_i
        # torch.manual_seed(seed)

        max_positions_train = (args.fixed_max_len, args.fixed_max_len)

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

        # reset meters
        for key, val in g_logging_meters.items():
            if val is not None:
                val.reset()
        for key, val in d_logging_meters.items():
            if val is not None:
                val.reset()

        # set training mode
        generator.train()
        discriminator.train()
        update_learning_rate(num_update, 8e4, args.g_learning_rate, args.lr_shrink, g_optimizer)

        for i, sample in enumerate(itr):
            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            ## part I: use gradient policy method to train the generator

            # use policy gradient training when rand > 50%
            rand = random.random()
            if rand >= 0.5:
                # policy gradient training
                generator.decoder.is_testing = True
                sys_out_batch, prediction = generator(sample)
                generator.decoder.is_testing = False
                with torch.no_grad():
                    reward = discriminator(sample['net_input']['src_tokens'], prediction, dataset.dst_dict.pad())
                train_trg_batch = sample['target']
                pg_loss = pg_criterion(sys_out_batch, train_trg_batch, reward, use_cuda)
                # logging.debug("G policy gradient loss at batch {0}: {1:.3f}, lr={2}".format(i, pg_loss.item(), g_optimizer.param_groups[0]['lr']))
                g_optimizer.zero_grad()
                pg_loss.backward()
                torch.nn.utils.clip_grad_norm(generator.parameters(), args.clip_norm)
                g_optimizer.step()

                # oracle valid
                sys_out_batch, _ = generator(sample)
                train_trg_batch = sample['target'].view(-1)
                sys_out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1))
                loss = g_criterion(sys_out_batch, train_trg_batch)
                sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
                logging_loss = loss.data / sample_size / math.log(2)
                g_logging_meters['train_loss'].update(logging_loss, sample_size)
                logging.debug("G MLE loss at batch {0}: {1:.3f}, lr={2}".format(i, g_logging_meters['train_loss'].avg,
                                                                                g_optimizer.param_groups[0]['lr']))
            else:
                # MLE training
                sys_out_batch, _ = generator(sample)
                train_trg_batch = sample['target'].view(-1)
                sys_out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1))
                loss = g_criterion(sys_out_batch, train_trg_batch)
                sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
                nsentences = sample['target'].size(0)
                logging_loss = loss.data / sample_size / math.log(2)
                g_logging_meters['bsz'].update(nsentences)
                g_logging_meters['train_loss'].update(logging_loss, sample_size)
                logging.debug("G MLE loss at batch {0}: {1:.3f}, lr={2}".format(i, g_logging_meters['train_loss'].avg,
                                                                                           g_optimizer.param_groups[0]['lr']))
                g_optimizer.zero_grad()
                loss.backward()
                # all-reduce grads and rescale by grad_denom
                for p in generator.parameters():
                    if p.requires_grad:
                        p.grad.data.div_(sample_size)
                torch.nn.utils.clip_grad_norm(generator.parameters(), args.clip_norm)
                g_optimizer.step()
            num_update += 1


            # part II: train the discriminator
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

            if use_cuda:
                labels = labels.cuda()

            disc_out = discriminator(src_sentence, trg_sentence, dataset.dst_dict.pad())
            d_loss = d_criterion(disc_out, labels)
            acc = torch.sum(torch.round(disc_out).squeeze(1) == labels).float() / len(labels)
            d_logging_meters['train_acc'].update(acc)
            d_logging_meters['train_loss'].update(d_loss)
            # logging.debug("D training loss {0:.3f}, acc {1:.3f} at batch {2}: ".format(d_logging_meters['train_loss'].avg,
            #                                                                            d_logging_meters['train_acc'].avg,
            #                                                                            i))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()



        # validation
        # set validation mode
        generator.eval()
        discriminator.eval()
        # Initialize dataloader
        max_positions_valid = (args.fixed_max_len, args.fixed_max_len)
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

        # reset meters
        for key, val in g_logging_meters.items():
            if val is not None:
                val.reset()
        for key, val in d_logging_meters.items():
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

                # generator validation
                sys_out_batch, _ = generator(sample)
                dev_trg_batch = sample['target'].view(-1)
                sys_out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1))
                loss = g_criterion(sys_out_batch, dev_trg_batch)
                sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
                loss = loss / sample_size / math.log(2)
                g_logging_meters['valid_loss'].update(loss, sample_size)
                logging.debug("G dev loss at batch {0}: {1:.3f}".format(i, g_logging_meters['valid_loss'].avg))

                # discriminator validation
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

                if use_cuda:
                    labels = labels.cuda()

                disc_out = discriminator(src_sentence, trg_sentence, dataset.dst_dict.pad())
                d_loss = d_criterion(disc_out, labels)
                acc = torch.sum(torch.round(disc_out).squeeze(1) == labels).float() / len(labels)
                d_logging_meters['valid_acc'].update(acc)
                d_logging_meters['valid_loss'].update(d_loss)
                # logging.debug("D dev loss {0:.3f}, acc {1:.3f} at batch {2}".format(d_logging_meters['valid_loss'].avg,
                #                                                                     d_logging_meters['valid_acc'].avg, i))

        torch.save(generator,
                   open(checkpoints_path + "joint_{0:.3f}.epoch_{1}.pt".format(g_logging_meters['valid_loss'].avg, epoch_i),
                        'wb'), pickle_module=dill)

        if g_logging_meters['valid_loss'].avg < best_dev_loss:
            best_dev_loss = g_logging_meters['valid_loss'].avg
            torch.save(generator, open(checkpoints_path + "best_gmodel.pt", 'wb'), pickle_module=dill)


def update_learning_rate(update_times, target_times, init_lr, lr_shrink, optimizer):
    lr = init_lr * (lr_shrink ** (update_times // target_times))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)