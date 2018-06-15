import logging
import math
import os
from collections import OrderedDict

import torch
from torch import cuda
import torch.nn.functional as F

import utils
from meters import AverageMeter
from generator import RNNModel
# from generator2 import RNNModel


def train_g(args, dataset):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

    use_cuda = (cuda.device_count() >= 1)

    # check checkpoints saving path
    if not os.path.exists('checkpoints/generator'):
        os.makedirs('checkpoints/generator')

    checkpoints_path = 'checkpoints/generator/'

    logging_meters = OrderedDict()
    logging_meters['train_loss'] = AverageMeter()
    logging_meters['valid_loss'] = AverageMeter()
    logging_meters['bsz'] = AverageMeter()  # sentences per batch
    logging_meters['update_times'] = AverageMeter()

    # Build model
    generator = RNNModel(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)

    if use_cuda:
        if torch.cuda.device_count() > 1:
            generator = torch.nn.DataParallel(generator).cuda()
        else:
            generator.cuda()
    else:
        generator.cpu()

    optimizer = eval("torch.optim." + args.g_optimizer)(generator.parameters(), args.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=args.lr_shrink)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf

    epoch_i = 1
    best_dev_loss = math.inf
    lr = optimizer.param_groups[0]['lr']
    # main training loop
    while lr > args.min_g_lr and epoch_i <= max_epoch:
        logging.info("At {0}-th epoch.".format(epoch_i))

        seed = args.seed + epoch_i
        torch.manual_seed(seed)

        max_positions_train = (args.fixed_max_len, args.fixed_max_len)

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
        for key, val in logging_meters.items():
            if val is not None:
                val.reset()

        for i, sample in enumerate(itr):

            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            train_trg_batch = sample['target'].view(-1)
            sys_out_batch = generator(sample)
            loss = F.nll_loss(sys_out_batch, train_trg_batch, size_average=False, ignore_index=dataset.dst_dict.pad(), reduce=True)
            sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
            nsentences = sample['target'].size(0)
            logging_loss = loss.item() / sample_size / math.log(2)
            logging_meters['bsz'].update(nsentences)
            logging_meters['train_loss'].update(logging_loss, sample_size)
            logging.debug(
                "g loss at batch {0}: {1:.3f}, batch size: {2}, lr={3}".format(i, logging_meters['train_loss'].avg,
                                                                             round(logging_meters['bsz'].avg),
                                                                             optimizer.param_groups[0]['lr']))
            optimizer.zero_grad()
            loss.backward()

            # all-reduce grads and rescale by grad_denom
            for p in generator.parameters():
                if p.requires_grad:
                    p.grad.data.div_(sample_size)

            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            optimizer.step()

            del sys_out_batch, loss

        # validation -- this is a crude estimation because there might be some padding at the end
        max_positions_valid = (args.fixed_max_len, args.fixed_max_len)

        # Initialize dataloader
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
        # set validation mode
        generator.eval()

        # reset meters
        for key, val in logging_meters.items():
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
                logging_loss = loss.item() / sample_size / math.log(2)
                logging_meters['valid_loss'].update(logging_loss, sample_size)
                logging.debug("g dev loss at batch {0}: {1:.3f}".format(i, logging_meters['valid_loss'].avg))

                del sys_out_batch, loss

                # update learning rate
        lr_scheduler.step(logging_meters['valid_loss'].avg)
        lr = optimizer.param_groups[0]['lr']

        logging.info(
            "Average g loss value per instance is {0} at the end of epoch {1}".format(logging_meters['valid_loss'].avg,
                                                                                    epoch_i))
        torch.save(generator.state_dict(), open(
            checkpoints_path + "data.nll_{0:.3f}.epoch_{1}.pt".format(logging_meters['valid_loss'].avg, epoch_i), 'wb'))

        if logging_meters['valid_loss'].avg < best_dev_loss:
            best_dev_loss = logging_meters['valid_loss'].avg
            torch.save(generator.state_dict(), open(checkpoints_path + "best_gmodel.pt", 'wb'))

        epoch_i += 1


