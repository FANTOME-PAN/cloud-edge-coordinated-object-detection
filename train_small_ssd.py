from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd_small import build_small_ssd, SmallSSD
from ssd import SSD, build_ssd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
from train_big_ssd import adjust_learning_rate, weights_init
from data.config import helmet_lite


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='helmet', choices=['VOC', 'COCO', 'helmet'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=HELMET_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default=None,
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


def train(path):
    if args.dataset == 'COCO':
        raise NotImplementedError()
    elif args.dataset == 'VOC':
        raise NotImplementedError()
    elif args.dataset == 'helmet':
        cfg = helmet_lite
        dataset = HelmetDetection(root=HELMET_ROOT, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    else:
        raise RuntimeError()

    ssd_net = build_small_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net
    big_net = build_ssd('train', helmet['min_dim'], helmet['num_classes']).cuda()
    big_net.load_state_dict(torch.load(path))

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        if args.basenet is None:
            print('Training vgg net')
            train_vgg(ssd_net.cuda(), big_net, data_loader)
        else:
            init_ssd_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network...')
            ssd_net.load_state_dict(init_ssd_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    net.train()

    print('Start training...')
    train_detection_net(net, big_net, data_loader, cfg, True)


def train_vgg(net: SmallSSD, big_net: SSD, data_loader: data.DataLoader, cfg=helmet_lite):
    print('---- training vgg net ----')
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    net.train()

    # loss counters
    total_loss = 0.

    step_index = 0

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()

        t0 = time.time()
        y_vgg_lite = net.vgg_forward(images)
        with torch.no_grad():
            y_vgg16 = big_net.vgg_forward(images)
        # back prop
        optimizer.zero_grad()
        loss = loss_fn(y_vgg_lite, y_vgg16)
        loss.backward()
        optimizer.step()
        t1 = time.time()
        total_loss += loss.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % loss.item(), end=' ')

        if iteration != 0 and iteration % 500 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), 'weights/small_ssd_vgg_' +
                       repr(iteration) + '.pth')
    torch.save(net.state_dict(),
               args.save_folder + 'small_ssd_vgg_' + args.dataset + '.pth')
    pass


def train_detection_net(net: SmallSSD, big_net: SSD, data_loader: data.DataLoader, cfg, distill=True):
    print('---- training detection net ----')
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda, knowledge_distill=distill)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0

    step_index = 0

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]

        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        if distill:
            with torch.no_grad():
                big_pred = big_net(images)
            loss_l, loss_c, loss_c_2 = criterion(out, targets, big_pred)
            loss = loss_l + (loss_c + loss_c_2) * 0.5
        else:
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

        if iteration != 0 and iteration % 1000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), 'weights/ttsmall_ssd_helmet_' +
                       repr(iteration) + '.pth')
    torch.save(net.state_dict(),
               args.save_folder + 'ttsmall_ssd_' + args.dataset + '.pth')


if __name__ == '__main__':
    train('weights/ssd300_helmet_4000.pth')
