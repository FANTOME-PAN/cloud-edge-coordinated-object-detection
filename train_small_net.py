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
from small_net import SmallNet, ConfNet1
from utils.evaluations import parse_rec, voc_ap, get_conf_gt
import os.path as osp


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


annopath = osp.join('%s', 'Annotations', '%s.xml')

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='helmet', choices=['VOC', 'COCO', 'helmet'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=HELMET_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--bignet', default='helmet_big_net.pth',
                    help='Big net for knowledge distilling')
parser.add_argument('--conf_dataset', default=None,
                    help='The dataset for training confidence net')
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


def train(skip_det_net=False):
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
    big_net.load_state_dict(torch.load(args.save_folder + args.bignet))
    conf_net = ConfNet1()
    small_net = SmallNet(ssd_net, conf_net)

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
            print('Training base net (vgg lite & conv6 conv7)')
            train_vgg(ssd_net.cuda(), big_net, data_loader)
        else:
            init_ssd_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network...')
            net.load_state_dict(init_ssd_weights)

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
    if not skip_det_net:
        train_detection_net(net, big_net, data_loader, cfg, True)
        torch.save(ssd_net.state_dict(), args.save_folder + 'helmet_detection_net.pth')
    else:
        print('skip training detection net.')

    # prepare data for confidence net
    if args.conf_dataset is None:
        test_dataset = HelmetDetection(root=HELMET_ROOT,
                                       transform=BaseTransform(300, MEANS))
        conf_dataset = prepare_data_for_conf_net(small_net, test_dataset, skip_preprocess=False)
        torch.save(conf_dataset, args.save_folder + 'conf_dataset.pkl')
    else:
        conf_dataset = torch.load(args.save_folder + args.conf_dataset)

    conf_dloader = data.DataLoader(conf_dataset, args.batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=True,
                                   pin_memory=True)

    if args.cuda:
        net = torch.nn.DataParallel(conf_net)
        cudnn.benchmark = True

    net.train()
    train_confidence_net(net, conf_dloader)


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
    ts = time.time()
    for iteration in range(20000):
        if iteration in [15000, 17500, 20000]:
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

        if iteration != 0 and iteration % 5000 == 0:
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


# 当 mAP 大于 conf_thresh，将 GT 设为 1；否则，设为 0
def prepare_data_for_conf_net(net: SmallNet, dataset, conf_thresh=0.5, skip_preprocess=False):
    net.det_net.phase = 'test'
    num_images = len(dataset)
    all_boxes = [[np.array([]) for _ in range(num_images)]
                 for _ in range(len(HELMET_CLASSES) + 1)]
    x_tensor = []
    y_tensor = []
    if not skip_preprocess:
        num_str = str(num_images)
        for i in range(num_images):
            im, gt, h, w = dataset.pull_item(i)
            x = im.unsqueeze(0).cuda()

            with torch.no_grad():
                detections = net(x)
            x_tensor.append(net.det_net.conf_net_input.cpu())
            get_conf_gt(detections, h, w, annopath % dataset.ids[i])

            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[j][i] = cls_dets
            print('image %s / %s' % (str(i + 1).zfill(len(num_str)), num_str))

        import pickle
        with open('x_tensor.pkl', 'wb') as f:
            pickle.dump(x_tensor, f)
        print('x_tensor saved.')
        with open('all_boxes.pkl', 'wb') as f:
            pickle.dump(all_boxes, f)
        print('all_boxes saved.')

    # write voc results file
    if not skip_preprocess:
        det_file_dict = [[] for _ in range(len(HELMET_CLASSES))]
        for cls_idx, cls in enumerate(HELMET_CLASSES):
            for im_idx in range(len(dataset)):
                dets = all_boxes[cls_idx + 1][im_idx]
                if len(dets) == 0:
                    continue
                # index[1] is the file name. e.g. 000001
                for k in range(dets.shape[0]):
                    det_file_dict[cls_idx].append((im_idx, dets[k, -1],
                                                   *(dets[k, :4] + 1)))
        import pickle
        with open('det_file_dict.pkl', 'wb') as f:
            pickle.dump(det_file_dict, f)
        print('det_file_dict saved.')

    if skip_preprocess:
        import pickle
        with open('x_tensor.pkl', 'rb') as f:
            x_tensor = pickle.load(f)
        with open('all_boxes.pkl', 'rb') as f:
            all_boxes = pickle.load(f)
        with open('det_file_dict.pkl', 'rb') as f:
            det_file_dict = pickle.load(f)

    # eval
    for im_idx in range(len(dataset)):
        print('IMAGE ID %d:' % im_idx)
        aps = []
        for i, cls in enumerate(HELMET_CLASSES):
            rec, prec, ap = my_voc_eval_per_img(det_file_dict, i, dataset, im_idx)
            aps.append(ap)
            print('AP for {} = {:.4f}'.format(cls, ap))
        y_tensor.append(1. if np.mean([x for x in aps if x > -0.5]) > conf_thresh else 0.)
    x_tensor = torch.cat(x_tensor, dim=0)
    y_tensor = torch.tensor(y_tensor)
    return data.TensorDataset(x_tensor, y_tensor)


# def my_voc_eval(det_file_dict, cls_idx, dataset, ovthresh=0.5):
#     class_recs = [dict() for i in range(len(dataset))]
#     npos = 0
#     for i in range(len(dataset)):
#         im, gt = dataset[i]
#         R = [obj for obj in gt if obj[-1] == cls_idx]
#         bbox = np.array([obj[:4] for obj in R])
#         det = [False] * len(R)
#         npos += len(R)
#         class_recs[i] = {'bbox': bbox,
#                          'det': det}
#
#     det_file = det_file_dict[cls_idx]
#     if len(det_file) > 0:
#         image_ids = [x[0] for x in det_file]
#         confidence = np.array([x[1] for x in det_file])
#         BB = np.array([[z for z in x[2:]] for x in det_file])
#
#         # sort by confidence score
#         sorted_idx = np.argsort(-confidence)
#         # sorted_scores = np.sort(-confidence)
#         BB = BB[sorted_idx, :]
#         image_ids = [image_ids[x] for x in sorted_idx]
#
#         # mark TPs and FPs
#         nd = len(image_ids)
#         tp = np.zeros(nd)
#         fp = np.zeros(nd)
#         jmax = 999999
#         for d in range(nd):
#             R = class_recs[image_ids[d]]
#             bb = BB[d, :].astype(float)
#             ovmax = -np.inf
#             BBGT = R['bbox'].astype(float)
#             if BBGT.size > 0:
#                 # compute overlaps
#                 # intersection
#                 ixmin = np.maximum(BBGT[:, 0], bb[0])
#                 iymin = np.maximum(BBGT[:, 1], bb[1])
#                 ixmax = np.minimum(BBGT[:, 2], bb[2])
#                 iymax = np.minimum(BBGT[:, 3], bb[3])
#                 iw = np.maximum(ixmax - ixmin, 0.)
#                 ih = np.maximum(iymax - iymin, 0.)
#                 inters = iw * ih
#                 uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
#                        (BBGT[:, 2] - BBGT[:, 0]) *
#                        (BBGT[:, 3] - BBGT[:, 1]) - inters)
#                 overlaps = inters / uni
#                 ovmax = np.max(overlaps)
#                 jmax = np.argmax(overlaps)
#
#             if ovmax > ovthresh:
#                 if not R['det'][jmax]:
#                     tp[d] = 1.
#                     R['det'][jmax] = 1
#                 else:
#                     fp[d] = 1.
#             else:
#                 fp[d] = 1.
#
#         fp = np.cumsum(fp)
#         tp = np.cumsum(tp)
#         rec = tp / float(npos)
#
#         prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#         ap = voc_ap(rec, prec)
#     else:
#         rec = -1.
#         prec = -1.
#         ap = -1.
#     return rec, prec, ap
#

def my_voc_eval_per_img(det_file_dict, cls_idx, dataset, idx, ovthresh=0.5):
    # class_recs = [dict() for i in range(len(dataset))]
    npos = 0
    # for i in range(len(dataset)):
    #  dataset.ids[idx]
    rec = parse_rec(annopath % dataset.ids[idx])
    # im, gt = dataset[idx]
    R = [obj for obj in rec if obj['name'] == HELMET_CLASSES[cls_idx]]
    if len(R) == 0:
        return -1., -1., -1.
    bbox = np.array([obj['bbox'] for obj in R])
    det = [False] * len(R)
    npos += len(R)
    class_rec = {'bbox': bbox, 'det': det}
    # class_recs[idx] = {'bbox': bbox,
    #                  'det': det}

    det_file = det_file_dict[cls_idx]
    if len(det_file) > 0:
        image_ids = [x[0] for x in det_file if x[0] == idx]
        confidence = np.array([x[1] for x in det_file if x[0] == idx])
        BB = np.array([[z for z in x[2:]] for x in det_file if x[0] == idx])

        # sort by confidence score
        sorted_idx = np.argsort(-confidence)
        # sorted_scores = np.sort(-confidence)
        BB = BB[sorted_idx, :]
        image_ids = [image_ids[x] for x in sorted_idx]

        # mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        jmax = 999999
        for d in range(nd):
            # R = class_recs[image_ids[d]]
            R = class_rec
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)

        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
    else:
        rec = -1.
        prec = -1.
        ap = -1.
    return rec, prec, ap


def confidence_loss(y_pred, targets, neg_weight=3.):
    y = targets.view(y_pred.size())
    if y.dtype != torch.float:
        y = y.type(torch.float)
    # y_inverse[i] = not y[i]
    y_inv = -y + 1.
    # if y[i] == 1 then y_k[i] == 1. else -1.
    y_k = y - y_inv
    # if y[i] == 1 then y_neg[i] == 1. else neg_weight
    y_neg = y + neg_weight * y_inv
    ret = -torch.log(y_pred * y_k + y_inv) * y_neg
    return ret.mean()


def get_correct_with_confidence(out_con: torch.Tensor, y_con: torch.Tensor, con_thresh=0.5, as_tensor=False):
    con_over_thresh = out_con > con_thresh
    pred_right = y_con.type(torch.uint8).view(-1, 1)
    assert pred_right.shape == con_over_thresh.shape
    correct = (pred_right * con_over_thresh).sum().item()
    total = con_over_thresh.sum().item()

    con = out_con > 0.5
    tmp = con == pred_right
    con_correct = tmp.sum().item()
    con_total = con.size(0)
    correct_1 = (tmp * pred_right).sum().item()
    correct_2 = con_correct - correct_1
    total_1 = pred_right.sum().item()
    total_2 = con_total - total_1
    details = (correct_1, total_1, correct_2, total_2)
    # correct and total within confidence; number of correct confidence and total confident case.
    if as_tensor:
        ret = torch.tensor([correct, total, con_correct, con_total, *details], dtype=torch.float).cuda()
        return ret
    return correct, total, con_correct, con_total, details


def train_confidence_net(net, data_loader: data.DataLoader):
    print('---- training confidence net ----')
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(200):
        # init
        total_loss = 0.
        correct = 0.
        total = 0.
        con_correct = 0.
        con_total = 0.

        # train
        for i, (x, y) in enumerate(data_loader):
            x = x.cuda()
            y = y.cuda()

            y_pred = net(x)
            loss = confidence_loss(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tp1, tp2, tp3, tp4, _ = get_correct_with_confidence(y_pred, y)

            total_loss += loss.item()
            correct += tp1
            total += tp2
            con_correct += tp3
            con_total += tp4

        # eval
        precision = correct / total if total > 0 else 1.
        recall = con_correct / con_total
        upload = 1. - total / con_total
        print('Epoch %2d:\ntrain loss= %7.4f, precision= %.4f, recall= %.4f, upload= %.4f' % (
            epoch + 1, total_loss, precision, recall, upload
        ))
    torch.save(net.state_dict(), args.save_folder + 'detection_net.pth')


if __name__ == '__main__':
    train(True)
