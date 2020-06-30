def compute_next_size(input_size, kernel_size, stride=1, padding=0, dilation=0):
    kernel_size += dilation * (kernel_size - 1)
    return (input_size - kernel_size + 2 * padding) // stride + 1


def eval_conv(in_size, in_chnl, out_chnl, kernel_size, stride=1, padding=0, dilation=0):
    out_size = compute_next_size(in_size, kernel_size, stride, padding, dilation)
    cal_per_elem = 2 * kernel_size * kernel_size * in_chnl
    ret = cal_per_elem * out_size * out_size * out_chnl
    return ret, out_size


def eval_fc(in_features, out_features):
    return 2 * in_features * out_features


def eval_pooling(in_size, in_chnl, kernel_size=2, stride=2, padding=0):
    size = compute_next_size(in_size, kernel_size, stride, padding)
    return size * size * in_chnl * kernel_size * kernel_size, size


def eval_big_net():
    params = [0, 28]

    def block(in_chnl, out_chnl):
        cal, size = eval_conv(params[1], in_chnl, out_chnl, 3)
        size = compute_next_size(size, 2, 2)
        params[0] += cal
        params[1] = size

    block(1, 32)
    block(32, 64)
    block(64, 128)

    params[0] += eval_fc(128, 625)
    params[0] += eval_fc(625, 10)
    print('big net FLOP: %d' % params[0])
    return params[0]


def vgg16_conv(size=300):
    print('---- VGG 16 evaluation ----')
    total_flop = 0
    # block 1
    flop, size = eval_conv(size, 3, 64, kernel_size=3, stride=1, padding=1)
    tmp, size = eval_conv(size, 64, 64, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 64, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk1: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 2
    flop = 0
    tmp, size = eval_conv(size, 64, 128, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 128, 128, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 128, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk2: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 3
    flop = 0
    tmp, size = eval_conv(size, 128, 256, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 256, 256, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 256, 256, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 256, kernel_size=2, stride=2, padding=1)
    flop += tmp
    print('blk3: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 4
    flop = 0
    tmp, size = eval_conv(size, 256, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 512, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk4: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 5
    flop = 0
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    print('blk5: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    print('vgg16 conv layers: total flop= %s' % format(total_flop, ','))
    return total_flop


def vgg11_conv(size=300):
    print('---- VGG 11 evaluation ----')
    total_flop = 0
    # block 1
    flop = 0
    tmp, size = eval_conv(size, 3, 64, kernel_size=3, stride=1, padding=1)
    # tmp, size = eval_conv(size, 64, 64, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 64, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk1: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 2
    flop = 0
    tmp, size = eval_conv(size, 64, 128, kernel_size=3, stride=1, padding=1)
    flop += tmp
    # tmp, size = eval_conv(size, 128, 128, kernel_size=3, stride=1, padding=1)
    # flop += tmp
    tmp, size = eval_pooling(size, 128, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk2: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 3
    flop = 0
    tmp, size = eval_conv(size, 128, 256, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 256, 256, kernel_size=3, stride=1, padding=1)
    flop += tmp
    # tmp, size = eval_conv(size, 256, 256, kernel_size=3, stride=1, padding=1)
    # flop += tmp
    tmp, size = eval_pooling(size, 256, kernel_size=2, stride=2, padding=1)
    flop += tmp
    print('blk3: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 4
    flop = 0
    tmp, size = eval_conv(size, 256, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    # tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    # flop += tmp
    tmp, size = eval_pooling(size, 512, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk4: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 5
    flop = 0
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    # tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    # flop += tmp
    print('blk5: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    print('vgg11 conv layers: total flop= %s' % format(total_flop, ','))
    return total_flop


def vgg11_reduced(size=300):
    print('---- VGG 11 reduced evaluation ----')
    total_flop = 0
    # block 1
    flop = 0
    tmp, size = eval_conv(size, 3, 64, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 64, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk1: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 2
    flop = 0
    tmp, size = eval_conv(size, 64, 128, kernel_size=3, stride=2, padding=1)
    flop += tmp
    # tmp, size = eval_pooling(size, 128, kernel_size=2, stride=2, padding=0)
    # flop += tmp
    print('blk2: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 3
    flop = 0
    tmp, size = eval_conv(size, 128, 256, kernel_size=3, stride=2, padding=1)
    flop += tmp
    # tmp, size = eval_pooling(size, 256, kernel_size=2, stride=2, padding=1)
    # flop += tmp
    print('blk3: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 4
    flop = 0
    tmp, size = eval_conv(size, 256, 512, kernel_size=3, stride=2, padding=1)
    flop += tmp
    # tmp, size = eval_pooling(size, 512, kernel_size=2, stride=2, padding=0)
    # flop += tmp
    print('blk4: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 5
    flop = 0
    # tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    # flop += tmp
    # print('blk5: flop = %15s || size = %3d' % (format(flop, ','), size))
    # total_flop += flop
    print('vgg11 conv layers: total flop= %s' % format(total_flop, ','))
    return total_flop


def conv6_7():
    print('---- conv6 and conv7 evaluation ----')
    flop = 0
    tmp, size = eval_pooling(19, 512, 3, 1, 1)
    flop += tmp
    tmp, size = eval_conv(19, 512, 1024, 3, padding=6, dilation=6)
    flop += tmp
    print('conv6 flop = %s' % format(tmp, ','))
    tmp, size = eval_conv(size, 1024, 1024, 1, 1, 0)
    flop += tmp
    print('conv7 flop = %s' % format(tmp, ','))
    print('total flop = %s' % format(flop, ','))
    return flop


def conv6_7_reduced():
    print('---- conv6 and conv7 reduced evaluation ----')
    flop = 0
    tmp, size = eval_pooling(19, 512, 3, 1, 1)
    flop += tmp
    tmp, size = eval_conv(19, 512, 512, 3, padding=6, dilation=6)
    flop += tmp
    print('conv6 flop = %s' % format(tmp, ','))
    tmp, size = eval_conv(size, 512, 1024, 1, 1, 0)
    flop += tmp
    print('conv7 flop = %s' % format(tmp, ','))
    print('total flop = %s' % format(flop, ','))
    return flop


def extra_feature_layers():
    print('---- extra feature layers evaluation ----')
    total_flop = 0

    name = 'Conv8_2'
    flop = 0
    _, size = eval_conv(19, 1024, 256, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 256, 512, 3, 2, 1)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv9_2'
    flop = 0
    _, size = eval_conv(size, 512, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 2, 1)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv10_2'
    flop = 0
    _, size = eval_conv(size, 256, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 1, 0)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv11_2'
    flop = 0
    _, size = eval_conv(size, 256, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 1, 0)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))
    print('total flop = %s' % format(total_flop, ','))
    return total_flop


def extra_feature_layers_reduced():
    print('---- extra feature layers reduced evaluation ----')
    total_flop = 0

    name = 'Conv8_2'
    flop = 0
    _, size = eval_conv(19, 1024, 256, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 256, 512, 3, 2, 1)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv9_2'
    flop = 0
    _, size = eval_conv(size, 512, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 2, 1)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv10_2'
    flop = 0
    _, size = eval_conv(size, 256, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 1, 0)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv11_2'
    flop = 0
    _, size = eval_conv(size, 256, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 1, 0)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))
    print('total flop = %s' % format(total_flop, ','))
    return total_flop


def classifiers(classes=5):
    print('---- classifiers evaluation ----')
    total = 0
    lst = [(38, 512, 4), (19, 1024, 6), (10, 512, 6), (5, 256, 6),
           (3, 256, 4), (1, 256, 4)]
    for i, (size, channels, prior_num) in enumerate(lst):
        flop = eval_conv(size, channels, prior_num * classes, 3, 1, 1)[0]
        flop += eval_conv(size, channels, prior_num * 4, 3, 1, 1)[0]
        total += flop
        print('Classifier %d: flop = %12s' % (i + 1, format(flop, ',')))
    print('total = %s' % format(total, ','))
    return total


def classifiers_reduced(classes=5):
    print('---- classifiers reduced evaluation ----')
    total = 0
    lst = [(19, 1024, 6), (10, 512, 6), (5, 256, 6),
           (3, 256, 4), (1, 256, 4)]
    for i, (size, channels, prior_num) in enumerate(lst):
        flop = eval_conv(size, channels, prior_num * classes, 3, 1, 1)[0]
        flop += eval_conv(size, channels, prior_num * 4, 3, 1, 1)[0]
        total += flop
        print('Classifier %d: flop = %12s' % (i + 1, format(flop, ',')))
    print('total = %s' % format(total, ','))
    return total


def conf_net():
    print('---- Confidence Net Evaluation ----')
    total = 0
    p = 5  # the bbox output may not be useful
    total += eval_fc(5 * 5 * 6 * p, 256)
    total += eval_fc(3 * 3 * 4 * p, 256)
    total += eval_fc(1 * 1 * 4 * p, 256)

    total += eval_fc(3 * 256, 2)
    print('flop = %s' % format(total, ','))


big = vgg16_conv()
big += conv6_7()
big += extra_feature_layers()
big += classifiers(5)
# vgg11_conv()
small = vgg11_reduced()
small += conv6_7_reduced()
small += extra_feature_layers_reduced()
small += classifiers_reduced(5)
print('\n\nBig net flop = %s\nSmall net flop = %s\n ratio = %f' % (format(big, ','), format(small, ','),
                                                                   small / big))
conf_net()
