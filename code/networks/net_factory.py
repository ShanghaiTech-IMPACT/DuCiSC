from networks.VNet import VNet, DuCiSC

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "DuCiSC" and mode == "train":
        net = DuCiSC(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "DuCiSC" and mode == "test":
        net = DuCiSC(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net
