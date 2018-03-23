from __future__ import print_function

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
from generator import NMT
import options
import data
from meters import AverageMeter
import numpy as np

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Driver program for JHU Adversarial-NMT.")

# Load args
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_checkpoint_args(parser, inference=True)
options.add_distributed_training_args(parser)


def main(args):

  use_cuda = (len(args.gpuid) >= 1)
  if args.gpuid:
    cuda.set_device(args.gpuid[0])

    # Load dataset
    splits = ['test']
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

  max_positions = 1e5

  itr = dataset.eval_dataloader(
    'test',
    max_sentences=args.max_sentences,
    max_positions=max_positions,
    skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test,
  )

  generator = torch.load(args.model_file)
  generator.decoder.is_testing = True
  generator.eval()

  if use_cuda > 0:
    generator.cuda()
  else:
    generator.cpu()

  with open('translation.txt', 'wb') as f_write:
    for i, sample in enumerate(itr):
      if use_cuda:
        sample['id'] = sample['id'].cuda()
        sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].cuda()
        sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].cuda()
        sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].cuda()
        sample['target'] = sample['target'].cuda()

      with torch.no_grad():
        _, prediction = generator(sample)  # (trg_seq_len, batch_size, trg_vocab_size)
        prediction = prediction.squeeze(0)

      # get translation sentence (without replacing bpe symbol)
      target_str = dataset.dst_dict.string(prediction, bpe_symbol=None, escape_unk=True)
      target_str += '\n'

      f_write.write(target_str.encode('utf-8'))
      print("{0:.3f}% is translated...".format(float(i + 1) * 100 / len(itr)))


if __name__ == "__main__":
  ret = parser.parse_known_args()
  args = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(args)