import argparse
import dill
import logging
import math
import os
import options
import random
from collections import OrderedDict

import torch
from torch import cuda
from torch.autograd import Variable

import data
from meters import AverageMeter
from generator import NMT
from discriminator import Discriminator
from train_generator import train_g
from train_discriminator import train_d


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

    g_criterion = torch.nn.NLLLoss(size_average=False, ignore_index=dataset.dst_dict.pad(),
                                 reduce=True)
    d_criterion = torch.nn.CrossEntropyLoss()

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
        # set training mode
        generator.train()

        # reset meters
        for key, val in g_logging_meters.items():
            if val is not None:
                val.reset()
        for key, val in d_logging_meters.items():
            if val is not None:
                val.reset()

        # srange generates a lazy sequence of shuffled range
        for i, sample in enumerate(itr):
            if use_cuda:
                sample['id'] = sample['id'].cuda()
                sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].cuda()
                sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].cuda()
                sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].cuda()
                sample['target'] = sample['target'].cuda()

            # part I: train generator as usual (but do NOT do bp to update weights)
            sys_out_batch, predictions = generator(sample)  # (trg_seq_len, batch_size, trg_vocab_size)
            train_trg_batch = sample['target'].view(-1)
            sys_out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1))
            g_loss = g_criterion(sys_out_batch, train_trg_batch)
            sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
            nsentences = sample['target'].size(0)
            logging_loss = g_loss.data / sample_size / math.log(2)
            g_logging_meters['bsz'].update(nsentences)
            g_logging_meters['train_loss'].update(logging_loss, sample_size)
            logging.debug("Generator loss at batch {0}: {1:.3f}".format(i, g_logging_meters['train_loss'].avg))

            # part II: discriminator judge predictions
            # prepare data for discriminator
            src_sentence = sample['net_input']['src_tokens']
            # train discriminator with human translation with probability 50%
            rand = random.random()
            if rand >= 0.5:
                trg_sentence = sample['net_input']['target']
                labels = Variable(torch.ones(sample['target'].size(0)).long())
            else:
                trg_sentence = Variable(predictions.data, requires_grad=True)
                labels = Variable(torch.zeros(sample['target'].size(0)).long())

            padded_src_sentence = pad_sentences(src_sentence, dataset.dst_dict.pad())
            padded_trg_sentence = pad_sentences(trg_sentence, dataset.dst_dict.pad())
            padded_src_embed = generator.encoder.embed_tokens(padded_src_sentence)
            padded_trg_embed = generator.decoder.embed_tokens(padded_trg_sentence)

            # build 2D-image like tensor
            src_temp = torch.stack([padded_src_embed] * 50, dim=0)
            trg_temp = torch.stack([padded_trg_embed] * 50, dim=0)
            disc_input = torch.cat([src_temp, trg_temp], dim=1)

            disc_out = discriminator(disc_input)
            labels = Variable(torch.ones(sample['target'].size(0)).long())
            if use_cuda:
                labels = labels.cuda()

            d_loss = d_criterion(disc_out, labels)
            logging.debug("Discriminator training loss at batch {0}: {1:.3f}".format(i, loss.item()))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # if we train discriminator with machine translation
            # we reward the NMT with probability 50%
            if rand < 0.5:
                if random.random() >= 0.5:
                    reward = math.log(1 - disc_out[1])
                else:
                    reward = 1.0

            g_optimizer.zero_grad()
            g_loss.backward()
            # all-reduce grads and rescale by grad_denom
            for p in generator.parameters():
                if p.requires_grad:
                    p.grad.data.div_(sample_size)
            torch.nn.utils.clip_grad_norm(generator.parameters(), args.clip_norm)
            g_optimizer.step()

        # validation -- this is a crude estimation because there might be some padding at the end
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


def pad_sentences(sentences, pad_idx, size=50):
    res = sentences[0].new(len(sentences), size).fill_(pad_idx)
    for i, v in enumerate(sentences):
        res[i][:, len(v)] = v
    return res

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)