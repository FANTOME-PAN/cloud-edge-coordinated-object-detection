## **训练计划 01:**

#### **1. train big ssd net**

如题

`python train_big_ssd.py`

#### **2. train vgg (in small ssd, 下同)**

用 big net 的 vgg 的输出作为 ground truth,
在 ImageNet 数据集上训练 vgg 。
该步骤为预训练，不确定需要训练到何种程度。

#### **3. train detection net**

用 knowledge distilling 训练 small ssd 中的检测网络。

#### **4. prepare data set**

准备用于 confidence net 训练的数据集。

根据训练好的 detection net 的输出与 ground truth 比对，得到 mAP 作为指标。
如果 mAP 大于一定阈值，则认为该图片易于识别；否则，认为其难以识别。
将易于识别的图片的 GT 设为 1，难以识别设为 0

#### **5. train confidence net**

用第 4 步准备的数据集，训练 confidence net
