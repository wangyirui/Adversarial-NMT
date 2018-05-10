import argparse
import logging
import math
import os
import options
from collections import OrderedDict

import torch
from torch import cuda
import torch.nn.functional as F

import data
import utils
from meters import AverageMeter
from discriminator import Discriminator
from generator import LSTMModel
from train_generator import train_g
from train_discriminator import train_d
from PGLoss import PGLoss

from disc_dataloader import DatasetProcessing, prepare_training_data
from disc_dataloader import train_dataloader, eval_dataloader



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
    print("{0} GPU(s) are available".format(cuda.device_count()))

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
    g_logging_meters['reward'] = AverageMeter()
    g_logging_meters['train_loss'] = AverageMeter()
    g_logging_meters['valid_loss'] = AverageMeter()
    g_logging_meters['train_acc'] = AverageMeter()
    g_logging_meters['valid_acc'] = AverageMeter()

    d_logging_meters = OrderedDict()
    d_logging_meters['train_loss'] = AverageMeter()
    d_logging_meters['valid_loss'] = AverageMeter()
    d_logging_meters['train_acc'] = AverageMeter()
    d_logging_meters['valid_acc'] = AverageMeter()

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
    discriminator.load_state_dict(model_dict)

    print("Discriminator has successfully loaded!")

    if use_cuda:
        if torch.cuda.device_count() > 1:
            discriminator = torch.nn.DataParallel(discriminator).cuda()
            generator = torch.nn.DataParallel(generator).cuda()
        else:
            generator.cuda()
            discriminator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()

    # adversarial training checkpoints saving path
    if not os.path.exists('checkpoints/joint'):
        os.makedirs('checkpoints/joint')
    checkpoints_path = 'checkpoints/joint/'

    # define loss function
    d_criterion = torch.nn.CrossEntropyLoss()
    pg_criterion = PGLoss(ignore_index=dataset.dst_dict.pad(), size_average=False,reduce=True)

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
    max_gan_epochs = 20

    best_d_dev_loss = float('inf')
    best_g_dev_loss = float('inf')

    # main training loop
    for epoch_i in range(1, max_gan_epochs + 1):
        # train generator
        seed = args.seed + 20 + epoch_i # avoid having same seed as pre-train
        torch.manual_seed(seed)

        max_positions_train = (args.fixed_max_len, args.fixed_max_len)

        itr = dataset.train_dataloader(
            'train',
            max_tokens=args.max_tokens,
            max_sentences=args.joint_batch_size,
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

        for i, sample in enumerate(itr):
            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            # step 1: train generator with PG loss
            logprobs = -generator(sample)
            disc_out   = discriminator(sample['net_input']['src_tokens'], sample['target'])
            reward = F.softmax(disc_out, dim=1)[:, 1]
            pg_loss = pg_criterion(logprobs, sample['target'], reward, use_cuda)
            g_optimizer.zero_grad()
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_norm)
            g_optimizer.step()

            # step 2: train generator with MLE
            train_trg_batch = sample['target'].view(-1)
            sys_out_batch = generator(sample)
            mle_loss = F.nll_loss(sys_out_batch, train_trg_batch, size_average=False, ignore_index=dataset.dst_dict.pad(), reduce=True)
            sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
            logging_loss = mle_loss.item() / sample_size / math.log(2)
            g_logging_meters['train_loss'].update(logging_loss, sample_size)
            logging.debug("g loss at batch {0}: {1:.3f}, lr={3}".format(i, g_logging_meters['train_loss'].avg,
                                                                        torch.mean(reward),
                                                                        g_optimizer.param_groups[0]['lr']))
            g_optimizer.zero_grad()
            mle_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_norm)
            g_optimizer.step()


        # validation
        # validation -- this is a crude estimation because there might be some padding at the end
        max_positions_valid = (args.fixed_max_len, args.fixed_max_len)

        # Initialize dataloader
        itr = dataset.eval_dataloader(
            'valid',
            max_tokens=args.max_tokens,
            max_sentences=args.prepare_dis_batch_size,
            max_positions=max_positions_valid,
            skip_invalid_size_inputs_valid_test=True,
            descending=True,  # largest batch first to warm the caching allocator
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )
        # set validation mode
        generator.eval()

        # reset meters
        for key, val in g_logging_meters.items():
            if val is not None:
                val.reset()
        with torch.no_grad():
            for i, sample in enumerate(itr):
                if use_cuda:
                    # wrap input tensors in cuda tensors
                    sample = utils.make_variable(sample, cuda=cuda)
                val_trg_batch = sample['target'].view(-1)
                sys_out_batch = generator(sample)
                loss = F.nll_loss(sys_out_batch, val_trg_batch, size_average=False, ignore_index=dataset.dst_dict.pad(), reduce=True)
                sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
                loss = loss.item() / sample_size / math.log(2)
                g_logging_meters['valid_loss'].update(loss, sample_size)
                logging.debug("g dev loss at batch {0}: {1:.3f}".format(i, g_logging_meters['valid_loss'].avg))


        logging.info(
            "Average g loss value per instance is {0} at the end of epoch {1}".format(
                g_logging_meters['valid_loss'].avg,
                epoch_i))
        torch.save(generator.state_dict(), open(
            checkpoints_path + "g.nll_{0:.3f}.epoch_{1}.pt".format(g_logging_meters['valid_loss'].avg, epoch_i),
            'wb'))

        if g_logging_meters['valid_loss'].avg < best_g_dev_loss:
            best_g_dev_loss = g_logging_meters['valid_loss'].avg
            torch.save(generator.state_dict(), open(checkpoints_path + "best_gmodel.pt", 'wb'))



        # train discriminator
        # sample is controlled by args.d_sample_without_replacement
        train = prepare_training_data(args, dataset, 'train', generator, epoch_i, use_cuda)
        valid = prepare_training_data(args, dataset, 'valid', generator, epoch_i, use_cuda)
        data_train = DatasetProcessing(data=train, maxlen=args.fixed_max_len)
        data_valid = DatasetProcessing(data=valid, maxlen=args.fixed_max_len)

        seed = args.seed + epoch_i
        torch.manual_seed(seed)

        # discriminator training dataloader
        train_loader = train_dataloader(data_train, batch_size=args.joint_batch_size,
                                        seed=seed, epoch=epoch_i, sort_by_source_size=False)

        valid_loader = eval_dataloader(data_valid, num_workers=4, batch_size=args.joint_batch_size)

        # set training mode
        discriminator.train()

        # set generator as eval
        generator.eval()

        # reset meters
        for key, val in d_logging_meters.items():
            if val is not None:
                val.reset()

        for i, sample in enumerate(train_loader):
            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=use_cuda)

            disc_out = discriminator(sample['src_tokens'], sample['trg_tokens'])

            loss = d_criterion(disc_out, sample['labels'])
            _, prediction = F.softmax(disc_out, dim=1).topk(1)
            acc = torch.sum(prediction == sample['labels'].unsqueeze(1)).float() / len(sample['labels'])

            d_logging_meters['train_acc'].update(acc.item())
            d_logging_meters['train_loss'].update(loss.item())
            logging.debug("D training loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ". \
                          format(d_logging_meters['train_loss'].avg, acc, d_logging_meters['train_acc'].avg,
                                 d_optimizer.param_groups[0]['lr'], i))

            d_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_norm)
            d_optimizer.step()

            # del src_tokens, trg_tokens, loss, disc_out, labels, prediction, acc
            del disc_out, loss, prediction, acc


        # validation
        # set validation mode
        discriminator.eval()
        with torch.no_grad():
            for i, sample in enumerate(valid_loader):
                with torch.no_grad():
                    if use_cuda:
                        # wrap input tensors in cuda tensors
                        sample = utils.make_variable(sample, cuda=use_cuda)

                    disc_out = discriminator(sample['src_tokens'], sample['trg_tokens'])

                    loss = d_criterion(disc_out, sample['labels'])
                    _, prediction = F.softmax(disc_out, dim=1).topk(1)
                    acc = torch.sum(prediction == sample['labels'].unsqueeze(1)).float() / len(sample['labels'])

                    d_logging_meters['valid_acc'].update(acc.item())
                    d_logging_meters['valid_loss'].update(loss.item())
                    logging.debug("D eval loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ". \
                                  format(d_logging_meters['valid_loss'].avg, acc, d_logging_meters['valid_acc'].avg,
                                         d_optimizer.param_groups[0]['lr'], i))

                del disc_out, loss, prediction, acc



        torch.save(discriminator.state_dict(), checkpoints_path + "d_ce_{0:.3f}_acc_{1:.3f}.epoch_{2}.pt" \
                       .format(d_logging_meters['valid_loss'].avg, d_logging_meters['valid_acc'].avg, epoch_i))

        if d_logging_meters['valid_loss'].avg < best_d_dev_loss:
            best_d_dev_loss = d_logging_meters['valid_loss'].avg
            torch.save(discriminator.state_dict(), checkpoints_path + "best_dmodel.pt")





if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)