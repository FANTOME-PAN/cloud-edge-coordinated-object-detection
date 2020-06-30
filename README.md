训练计划 01:

1. train big ssd net
如题

2. train vgg (in small ssd, 下同)
用 big net 的 vgg 的输出作为 ground truth,
在 ImageNet 数据集上训练 vgg 。
该步骤为预训练，不确定需要训练到何种程度。

3. train detection net
用 knowledge distilling 训练 small ssd 中的检测网络。

4. train confidence net
用 detection net 的输出结果与 ground truth 比对得到 conf net 的 ground truth，
训练 conf net
