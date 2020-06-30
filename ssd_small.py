import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from data import helmet_lite
import os


class SmallSSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes, cfg=helmet_lite):
        super(SmallSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources, loc, conf = [], [], []

        for v in self.vgg:
            x = v(x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(x.dtype)  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def vgg_forward(self, x):
        for v in self.vgg:
            x = v(x)
        return x

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_small_ssd(phase, size=300, num_classes=5):
    vgg = [
        nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(inplace=True),

        nn.MaxPool2d(3, 1, 1),  # is it necessary?
        nn.Conv2d(512, 512, 3, padding=6, dilation=6), nn.ReLU(inplace=True),
        nn.Conv2d(512, 1024, 1, 1, 0), nn.ReLU(inplace=True)
    ]
    extras = [
        nn.Conv2d(1024, 256, 1),
        nn.Conv2d(256, 512, 3, 2, 1),  # Conv8_2
        nn.Conv2d(512, 128, 1),
        nn.Conv2d(128, 256, 3, 2, 1),  # Conv9_2
        nn.Conv2d(256, 128, 1),
        nn.Conv2d(128, 256, 3),        # Conv10_2
        nn.Conv2d(256, 128, 1),
        nn.Conv2d(128, 256, 3),        # Conv11_2
    ]
    head = [
        # loc layers
        [
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        ],
        # conf layers
        [
            nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
        ]
    ]
    return SmallSSD(phase, size, vgg, extras, head, num_classes)




