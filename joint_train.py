import argparse
import logging
import math
import dill
import os
import options
import random
from collections import OrderedDict

import torch
from torch import cuda
from torch.autograd import Variable

import data
from meters import AverageMeter
from discriminator import Discriminator
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
options.add_model_args(parser)

def main(args):
    use_cuda = (len(args.gpuid) >= 1)
    if args.gpuid:
        cuda.set_device(args.gpuid[0])

    # Load dataset
    splits = ['train', 'valid']
    if data.has_binary_files(args.data, splits):
        dataset = data.load_dataset(
            args.data, splits, args.src_lang, args.trg_lang)
    else:
        dataset = data.load_raw_text_dataset(
            args.data, splits, args.src_lang, args.trg_lang)
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
    g_logging_meters['bsz'] = AverageMeter()  # sentences per batch
    g_logging_meters['update_times'] = AverageMeter()

    d_logging_meters = OrderedDict()
    d_logging_meters['train_loss'] = AverageMeter()
    d_logging_meters['valid_loss'] = AverageMeter()
    d_logging_meters['bsz'] = AverageMeter()  # sentences per batch
    d_logging_meters['update_times'] = AverageMeter()

    # try to load generator model
    g_model_path = 'checkpoints/generator/best_g_model.pt'
    if not os.path.exists(g_model_path):
        train_g(args, dataset)
    assert os.path.exists(g_model_path)
    generator = torch.load(g_model_path)

    # try to load discriminator model
    d_model_path = 'checkpoints/discriminator/best_d_model.pt'
    if not os._exists(d_model_path):
        train_d(args, dataset)
    assert  os.path.exists(d_model_path)
    discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    discriminator.load_state_dict(torch.load(d_model_path))

    # adversarial training checkpoints saving path
    if not os.path.exists('checkpoints/joint'):
        os.makedirs('checkpoints/joint')
    checkpoints_path = 'checkpoints/generator/'

    # define loss function
    g_criterion = torch.nn.NLLLoss(size_average=False, ignore_index=dataset.dst_dict.pad(),reduce=True)
    d_criterion = torch.nn.BCELoss()
    pg_criterion = PGLoss()

    # define optimizer
    g_optimizer = eval("torch.optim." + args.optimizer)(generator.parameters(), args.learning_rate)
    d_optimizer = eval("torch.optim." + args.optimizer)(discriminator.parameters(), args.learning_rate)

    # start joint training
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

        for i, sample in enumerate(itr):
            if use_cuda:
                sample['id'] = sample['id'].cuda()
                sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].cuda()
                sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].cuda()
                sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].cuda()
                sample['target'] = sample['target'].cuda()

            ## part I: use gradient policy method to train the generator

            # obtain generator output
            sys_out_batch, _ = generator(sample)
            # use policy gradient training with 50% probability
            rand = random.random()
            if rand >= 0.5:
                # obtain discriminator judge
                with torch.no_grad():
                    reward = discriminator(src_sentence, sys_out_batch, dataset.dst_dict.pad())
                train_trg_batch = sample['target']
                pg_loss = pg_criterion(sys_out_batch, train_trg_batch, reward)
                logging.debug("Generator policy gradient loss at batch {0}: {1:.3f}".format(i, pg_loss.item()))
                g_optimizer.zero_grad()
                pg_loss.backward()
                torch.nn.utils.clip_grad_norm(generator.parameters(), args.clip_norm)
                g_optimizer.step()
            else:
                train_trg_batch = sample['target'].view(-1)
                sys_out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1))
                loss = g_criterion(sys_out_batch, train_trg_batch)
                sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
                nsentences = sample['target'].size(0)
                logging_loss = loss.data / sample_size / math.log(2)
                g_logging_meters['bsz'].update(nsentences)
                g_logging_meters['train_loss'].update(logging_loss, sample_size)
                logging.debug("Generator loss at batch {0}: {1:.3f}".format(i, g_logging_meters['train_loss'].avg))
                g_optimizer.zero_grad()
                loss.backward()
                # all-reduce grads and rescale by grad_denom
                for p in generator.parameters():
                    if p.requires_grad:
                        p.grad.data.div_(sample_size)
                torch.nn.utils.clip_grad_norm(generator.parameters(), args.clip_norm)
                g_optimizer.step()

            # part II: train the discriminator

            src_sentence = sample['net_input']['src_tokens']
            # train discriminator with human translation with probability 50%
            rand = random.random()
            if rand >= 0.5:
                trg_sentence = sample['net_input']['target']
                labels = Variable(torch.ones(sample['target'].size(0)).long())
            else:
                with torch.no_grad():
                    sys_out_batch, _ = generator(sample)
                    _, trg_sentence = sys_out_batch.topk(1)
                labels = Variable(torch.zeros(sample['target'].size(0)).long())
            if use_cuda:
                labels = labels.cuda()
            disc_out = discriminator(src_sentence, trg_sentence, dataset.dst_dict.pad())
            d_loss = d_criterion(disc_out, labels)
            logging.debug("Discriminator training loss at batch {0}: {1:.3f}".format(i, d_loss.item()))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()


        # validation
        # set validation mode
        generator.eval()
        discriminator.eval()
        # Initialize dataloader
        max_positions_valid = (1e5, 1e5)
        itr = dataset.eval_dataloader(
            'valid',
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=max_positions_valid,
            skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test,
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
                logging.debug("Generator dev loss at batch {0}: {1:.3f}".format(i, g_logging_meters['valid_loss'].avg))

                # discriminator validation
                src_sentence = sample['net_input']['src_tokens']
                # valid discriminator with human translation with probability 50%
                rand = random.random()
                if rand >= 0.5:
                    trg_sentence = sample['net_input']['target']
                    labels = Variable(torch.ones(sample['target'].size(0)).long())
                else:
                    sys_out_batch, _ = generator(sample)
                    _, trg_sentence = sys_out_batch.topk(1)
                    labels = Variable(torch.zeros(sample['target'].size(0)).long())
                if use_cuda:
                    labels = labels.cuda()
                disc_out = discriminator(src_sentence, trg_sentence, dataset.dst_dict.pad())
                d_loss = d_criterion(disc_out, labels)
                logging.debug("Discriminator dev loss at batch {0}: {1:.3f}".format(i, d_loss.item()))

        torch.save(generator,
                   open(checkpoints_path + "joint_{0:.3f}.epoch_{1}.pt".format(g_logging_meters['valid_loss'].avg, epoch_i),
                        'wb'), pickle_module=dill)

        if g_logging_meters['valid_loss'].avg < best_dev_loss:
            best_dev_loss = g_logging_meters['valid_loss']
            torch.save(generator, open(checkpoints_path + "best_gmodel.pt", 'wb'), pickle_module=dill)


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)