import argparse
import codecs
import collections
import dill
import logging

import torch
import torchtext.vocab

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW5.")
parser.add_argument("--train_file", required=True,
                    help="Training text file that needs to be preprocessed.")
parser.add_argument("--dev_file", required=True,
                    help="Dev text file that needs to be preprocessed.")
parser.add_argument("--test_file", required=True,
                    help="Test text file that needs to be preprocessed.")
parser.add_argument("--vocab_file", required=True,
                    help="Torchtext vocab file that needs to be loaded.")
parser.add_argument("--data_file", required=True,
                    help="Path to store the binarized data file.")
parser.add_argument("--charniak", default=False, action='store_true',
                    help="Append BOS and EOS as in Charniak parser format.")

UNK = "<unk>"
PAD = "<blank>"
#BOS = "<s>"
EOS = "</s>"

def main(options):

  # vocab = torchtext.vocab.Vocab(token_cnt, max_size=options.vocab_size, specials=[PAD, BLK])
  itos = torch.load(open("data/" + options.vocab_file, 'rb'))
  # if options.charniak and BOS not in itos:
  #    itos.insert(0, BOS)
  if UNK not in itos:
    itos.insert(0, UNK)
  if EOS not in itos:
     itos.insert(0, EOS)
  if PAD not in itos:
    itos.insert(0, PAD)

  stoi = dict([(word, id) for (id, word) in enumerate(itos)])
  vocab = torchtext.vocab.Vocab(collections.Counter({}))
  vocab.itos = itos
  vocab.stoi = stoi
  unk_idx = stoi["<unk>"]

  train_data = []
  for line in codecs.open("data/"+options.train_file, 'r', 'utf8'):
    tokens = line.split()
    token_ids = []
    # if options.charniak:
    #   token_ids.append(vocab.stoi.get(BOS))
    for token in tokens:
      token_ids.append(vocab.stoi.get(token, unk_idx))
    # if options.charniak:
    #   token_ids.append(vocab.stoi.get(EOS))
    token_ids.append(vocab.stoi.get(EOS))
    sent = torch.LongTensor(token_ids)
    train_data.append(sent)

  dev_data = []
  for line in codecs.open("data/"+options.dev_file, 'r', 'utf8'):
    tokens = line.split()
    token_ids = []
    # if options.charniak:
    #   token_ids.append(vocab.stoi.get(BOS))
    for token in tokens:
      token_ids.append(vocab.stoi.get(token, unk_idx))
    # if options.charniak:
    #   token_ids.append(vocab.stoi.get(EOS))
    token_ids.append(vocab.stoi.get(EOS))
    sent = torch.LongTensor(token_ids)
    dev_data.append(sent)

  test_data = []
  for line in codecs.open("data/"+options.test_file, 'r', 'utf8'):
    tokens = line.split()
    token_ids = []
    # if options.charniak:
    #   token_ids.append(vocab.stoi.get(BOS))
    for token in tokens:
      token_ids.append(vocab.stoi.get(token, unk_idx))
    # if options.charniak:
    #   token_ids.append(vocab.stoi.get(EOS))
    token_ids.append(vocab.stoi.get(EOS))
    sent = torch.LongTensor(token_ids)
    test_data.append(sent)

  torch.save((train_data, dev_data, test_data, vocab), open(options.data_file, 'wb'), pickle_module=dill)


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
