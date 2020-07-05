import torch
from torch import nn
from ssd_small import SmallSSD


class ConfNet1(nn.Module):
    def __init__(self, num_classes=5):
        super(ConfNet1, self).__init__()
        n = num_classes
        self.conf1 = nn.Sequential(nn.Linear(5 * 5 * 6 * n, 256), nn.ReLU(inplace=True))
        self.conf2 = nn.Sequential(nn.Linear(3 * 3 * 4 * n, 256), nn.ReLU(inplace=True))
        self.conf3 = nn.Sequential(nn.Linear(1 * 1 * 4 * n, 256), nn.ReLU(inplace=True))
        self._split_pos = (5 * 5 * 6 * n + 3 * 3 * 4 * n + 1 * 1 * 4 * n,
                           3 * 3 * 4 * n + 1 * 1 * 4 * n,
                           1 * 1 * 4 * n)
        self.fc = nn.Sequential(nn.Linear(3 * 256, 1), nn.Sigmoid())

    # x is the conf output of the ssd
    def forward(self, x):
        p1, p2, p3 = self._split_pos
        # batch_num * 5*5 boxes, batch_num * 3*3 boxes, batch_num * 1*1 boxes
        x1, x2, x3 = x[:, -p1:-p2], x[:, -p2:-p3], x[:, -p3:]
        o1 = self.conf1(x1)
        o2 = self.conf2(x2)
        o3 = self.conf3(x3)
        out = torch.cat((o1, o2, o3), dim=1)
        out = self.fc(out)
        return out


class SmallNet(nn.Module):
    def __init__(self, det_net: SmallSSD, conf_net=ConfNet1()):
        super(SmallNet, self).__init__()
        self.det_net = det_net
        self.conf_net = conf_net
        self._input = None

    def forward(self, x):
        out = self.det_net(x)
        self._input = self.det_net.conf_net_input
        return out

    def conf(self, x=None):
        if x is not None:
            with torch.no_grad():
                self.forward(x)
        return self.conf_net(self._input)
