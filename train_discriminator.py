import logging
import random
import os
import math
import numpy as np
from collections import OrderedDict

import torch
from torch import cuda
from torch.autograd import Variable

import utils
from meters import AverageMeter
from discriminator import Discriminator
from generator import LSTMModel
from batch_generator import BatchGenerator

def train_d(args, dataset):
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

    # Load generator
    assert os.path.exists('checkpoints/generator/best_gmodel.pt')
    generator = LSTMModel(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    model_dict = generator.state_dict()
    pretrained_dict = torch.load('checkpoints/generator/best_gmodel.pt')
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    generator.load_state_dict(model_dict)

    if use_cuda:
        discriminator.cuda()
        generator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()

    criterion = torch.nn.CrossEntropyLoss()

    # optimizer = eval("torch.optim." + args.d_optimizer)(filter(lambda x: x.requires_grad, discriminator.parameters()),
                                                        # args.d_learning_rate, momentum=args.momentum, nesterov=True)
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, discriminator.parameters()), 1e-3)

    # Train until the accuracy achieve the define value
    max_epoch = args.max_epoch or math.inf
    epoch_i = 1
    trg_acc = 0.82
    best_dev_loss = math.inf

    # main training loop
    while epoch_i <= max_epoch:
        logging.info("At {0}-th epoch.".format(epoch_i))

        seed = args.seed + epoch_i
        torch.manual_seed(seed)

        # we must use a fixed max sentence length
        max_positions_train = (args.fixed_max_len, args.fixed_max_len)

        # Initialize dataloader, starting at batch_offset
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

        translator = BatchGenerator(
            generator, beam_size=args.beam, maxlen=args.fixed_max_len, stop_early=(not args.no_early_stop),
            normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
            unk_penalty=args.unkpen)

        # set training mode
        discriminator.train()

        # reset meters
        for key, val in logging_meters.items():
            if val is not None:
                val.reset()

        # training process
        for i, sample in enumerate(itr):
            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            # train D with fake translation and human translation
            # rand = random.random()
            # train D with ground-truth
            # if rand > 0.5:
            #     dis_input_tokens = sample['target']
            #     labels = Variable(torch.ones(sample['target'].size(0)).long())
            # # train D with machine translation
            # else:
            #     with torch.no_grad():
            #         dis_input_tokens = translator.generate_translation_tokens(sample,
            #                                                             beam_size=args.beam,
            #                                                             maxlen_a=args.max_len_a,
            #                                                             maxlen_b=args.max_len_b,
            #                                                             nbest=args.nbest)
            #         labels = Variable(torch.zeros(sample['target'].size(0)).long())

            # prepare true data
            true_tokens = sample['target']
            true_labels = Variable(torch.ones(sample['target'].size(0)).long())

            # prepare fake data
            with torch.no_grad():
                fake_tokens = translator.generate_translation_tokens(sample,
                                                                            beam_size=args.beam,
                                                                            maxlen_a=args.max_len_a,
                                                                            maxlen_b=args.max_len_b,
                                                                            nbest=args.nbest)
                fake_labels = Variable(torch.zeros(sample['target'].size(0)).long())

            bsz = true_labels.size(0)
            indices = np.random.permutation(2*bsz)
            dis_input_tokens = torch.cat([true_tokens, fake_tokens], dim=0)
            labels = torch.cat([true_labels, fake_labels], dim=0)
            dis_input_tokens = dis_input_tokens[indices]
            labels = labels[indices]
            dis_input_tokens = dis_input_tokens[:bsz]
            labels = labels[:bsz]

            src_tokens = sample['net_input']['src_tokens']
            disc_out = discriminator(src_tokens, dis_input_tokens, dataset.dst_dict.pad())

            if use_cuda:
                labels = labels.cuda()

            loss = criterion(disc_out, labels)
            _, prediction = disc_out.topk(1)
            acc = torch.sum(prediction.squeeze(1) == labels).float() / len(labels)
            logging_meters['train_acc'].update(acc)
            logging_meters['train_loss'].update(loss)
            logging.debug("D training loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(logging_meters['train_loss'].avg,
                                                                                                                acc,
                                                                                                               logging_meters['train_acc'].avg,
                                                                                                               optimizer.param_groups[0]['lr'],
                                                                                                               i,))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(discriminator.parameters(), args.clip_norm)
            optimizer.step()


        max_positions_valid = (args.fixed_max_len, args.fixed_max_len)
        # validation process
        # Initialize dataloader
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

        for i, sample in enumerate(itr):
            with torch.no_grad():
                if use_cuda:
                    # wrap input tensors in cuda tensors
                    sample = utils.make_variable(sample, cuda=cuda)

                # train D with fake translation and human translation
                rand = random.random()
                # train D with ground-truth
                if rand > 0.5:
                    dis_input_tokens = sample['target']
                    labels = Variable(torch.ones(sample['target'].size(0)).long())
                # train D with machine translation
                else:
                    dis_input_tokens = translator.generate_translation_tokens(sample,
                                                                        beam_size=args.beam,
                                                                        maxlen_a=args.max_len_a,
                                                                        maxlen_b=args.max_len_b,
                                                                        nbest=args.nbest)
                    labels = Variable(torch.zeros(sample['target'].size(0)).long())

                src_tokens = sample['net_input']['src_tokens']
                disc_out = discriminator(src_tokens, dis_input_tokens, dataset.dst_dict.pad())

                if use_cuda:
                    labels = labels.cuda()

                loss = criterion(disc_out, labels)
                _, prediction = disc_out.topk(1)
                acc = torch.sum(prediction.squeeze(1) == labels).float() / len(labels)
                logging_meters['valid_acc'].update(acc)
                logging_meters['valid_loss'].update(loss)
                logging.debug("D eval loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ".format(logging_meters['valid_loss'].avg,
                                                                                                                acc,
                                                                                                                logging_meters['valid_acc'].avg,
                                                                                                                optimizer.param_groups[0]['lr'],
                                                                                                                i))

        torch.save(discriminator.state_dict(), checkpoints_path + "ce_{0:.3f}.epoch_{1}.pt".format(logging_meters['valid_loss'].avg, epoch_i))

        if logging_meters['valid_loss'].avg < best_dev_loss:
            best_dev_loss = logging_meters['valid_loss'].avg
            torch.save(discriminator.state_dict(), checkpoints_path + "best_dmodel.pt")

        # pretrain the discriminator to achieve accuracy 82%
        if logging_meters['valid_acc'].avg >= trg_acc:
            return

        epoch_i += 1
