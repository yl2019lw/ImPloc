#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import sdataset
import tensorboardX
import os
import time
import numpy as np
from util import torch_util
from util import npmetrics
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def patch_collate(batch):
    imgs, targets = zip(*batch)
    nimgs = np.concatenate(imgs)
    ntargets = np.concatenate(targets)
    return torch.from_numpy(nimgs), torch.from_numpy(ntargets)


def train_simg(depth=18, size=3000):
    train_dataset = sdataset.SImgDataset("train", size)
    val_dataset = sdataset.SImgDataset("val", size)
    test_dataset = sdataset.SImgDataset("test", size)

    from sresnet import SResnet
    model = nn.DataParallel(SResnet(depth).cuda())
    dcfg = {"bsize": 2, "nworker": 8, "collate": default_collate}
    if size == 3000:
        dcfg['bsize'] = 2
        dcfg['nworker'] = 8
    if size == 224:
        dcfg['bsize'] = 256
        dcfg['nworker'] = 20

    model_name = "simg_res%d_size%d" % (depth, size)
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_spatch(depth=50, roiSize=100):
    train_dataset = sdataset.SPatchDataset("train", roiSize=roiSize)
    val_dataset = sdataset.SPatchDataset("val", roiSize=roiSize)
    test_dataset = sdataset.SPatchDataset("test", roiSize=roiSize)

    from sresnet import SResnet
    model = nn.DataParallel(SResnet(depth).cuda())
    dcfg = {"bsize": 8, "nworker": 8, "collate": patch_collate}

    model_name = "sp%d_res%d" % (roiSize, depth)
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_sallpatch(depth=50, roiSize=100):
    train_dataset = sdataset.SPatchAllDataset("train", roiSize=roiSize)
    val_dataset = sdataset.SPatchAllDataset("val", roiSize=roiSize)
    test_dataset = sdataset.SPatchAllDataset("test", roiSize=roiSize)

    from sresnet import SResnet
    model = nn.DataParallel(SResnet(depth).cuda())
    dcfg = {"bsize": 256, "nworker": 4, "collate": default_collate}

    model_name = "sp%dall_res%d" % (roiSize, depth)
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_sfpn(depth=50, roiSize=224):
    train_dataset = sdataset.SPatchDataset("train", roiSize=roiSize)
    val_dataset = sdataset.SPatchDataset("val", roiSize=roiSize)
    test_dataset = sdataset.SPatchDataset("test", roiSize=roiSize)

    from sfpn import SFPN
    model = nn.DataParallel(SFPN().cuda())
    dcfg = {"bsize": 6, "nworker": 6, "collate": patch_collate}

    model_name = "sp%d_sfpn%d" % (roiSize, depth)
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_sfpn_npatch(depth=50, roiSize=224):
    train_dataset = sdataset.SPatchDataset("train", roiSize=roiSize, npatch=2)
    val_dataset = sdataset.SPatchDataset("val", roiSize=roiSize, npatch=2)
    test_dataset = sdataset.SPatchDataset("test", roiSize=roiSize, npatch=2)

    from sfpn import SFPN
    model = nn.DataParallel(SFPN().cuda())
    dcfg = {"bsize": 64, "nworker": 20, "collate": patch_collate}

    model_name = "sp%d_sfpn%d_npatch2" % (roiSize, depth)
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_sfpn_pt(depth=50, roiSize=224):
    train_dataset = sdataset.DecodePatchDataset("train", roiSize=roiSize)
    val_dataset = sdataset.DecodePatchDataset("val", roiSize=roiSize)
    test_dataset = sdataset.DecodePatchDataset("test", roiSize=roiSize)

    from sfpn import SFPN
    model = nn.DataParallel(SFPN().cuda())
    dcfg = {"bsize": 192, "nworker": 20, "collate": default_collate}

    model_name = "sp%d_sfpn%d_decode" % (roiSize, depth)
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_sfpn_small(depth=50, roiSize=224):
    train_dataset = sdataset.SmallPatchDataset("train", roiSize=roiSize)
    val_dataset = sdataset.SmallPatchDataset("val", roiSize=roiSize)
    test_dataset = sdataset.SmallPatchDataset("test", roiSize=roiSize)

    from sfpn import SFPN
    model = nn.DataParallel(SFPN().cuda())
    # dcfg = {"bsize": 192, "nworker": 20, "collate": default_collate}
    dcfg = {"bsize": 64, "nworker": 20, "collate": default_collate}

    model_name = "sp%d_sfpn%d_small" % (roiSize, depth)
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_sfpn_small_balance(depth=50, roiSize=224):
    train_dataset = sdataset.SmallPatchDataset("train", roiSize, balance=True)
    val_dataset = sdataset.SmallPatchDataset("val", roiSize)
    test_dataset = sdataset.SmallPatchDataset("test", roiSize)

    from sfpn import SFPN
    model = nn.DataParallel(SFPN().cuda())
    dcfg = {"bsize": 192, "nworker": 20, "collate": default_collate}

    model_name = "sp%d_sfpn%d_small_balance" % (roiSize, depth)
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_sfpn_small_aug(depth=50, roiSize=224):
    top = torchvision.transforms.ToPILImage()
    hf = torchvision.transforms.RandomHorizontalFlip()
    vf = torchvision.transforms.RandomVerticalFlip()
    rot = torchvision.transforms.RandomRotation(30)
    size = torchvision.transforms.Resize((roiSize, roiSize))
    tot = torchvision.transforms.ToTensor()
    trfm = torchvision.transforms.Compose([top, hf, vf, rot, size, tot])

    train_dataset = sdataset.SmallPatchDataset(
        "train", roiSize=roiSize, transform=trfm)
    val_dataset = sdataset.SmallPatchDataset("val", roiSize=roiSize)
    test_dataset = sdataset.SmallPatchDataset("test", roiSize=roiSize)

    from sfpn import SFPN
    model = nn.DataParallel(SFPN().cuda())
    # dcfg = {"bsize": 192, "nworker": 20, "collate": default_collate}
    dcfg = {"bsize": 128, "nworker": 20, "collate": default_collate}

    model_name = "sp%d_sfpn%d_small_aug" % (roiSize, depth)
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_sfpn_shuffle(depth=50, roiSize=224):
    train_dataset = sdataset.ShufflePatchDataset("train", roiSize=roiSize)
    val_dataset = sdataset.ShufflePatchDataset("val", roiSize=roiSize)
    test_dataset = sdataset.ShufflePatchDataset("test", roiSize=roiSize)

    from sfpn import SFPN
    model = nn.DataParallel(SFPN().cuda())
    dcfg = {"bsize": 192, "nworker": 12, "collate": default_collate}

    model_name = "sp%d_sfpn%d_shuffle" % (roiSize, depth)
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_sallpatch_sfpn(depth=50, roiSize=224):
    train_dataset = sdataset.SPatchAllDataset("train", roiSize=roiSize)
    val_dataset = sdataset.SPatchAllDataset("val", roiSize=roiSize)
    test_dataset = sdataset.SPatchAllDataset("test", roiSize=roiSize)

    from sfpn import SFPN
    model = nn.DataParallel(SFPN().cuda())
    dcfg = {"bsize": 128, "nworker": 8, "collate": default_collate}

    model_name = "sp%dall_sfpn%d" % (roiSize, depth)
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_srdn(roiSize=32):
    train_dataset = sdataset.SmallPatchDataset("train", roiSize=roiSize)
    val_dataset = sdataset.SmallPatchDataset("val", roiSize=roiSize)
    test_dataset = sdataset.SmallPatchDataset("test", roiSize=roiSize)

    from srdn import RDN
    # model = nn.DataParallel(
    #   RDN(g0=32, d=4, c=6, k=16, roiSize=roiSize).cuda())
    model = nn.DataParallel(RDN(g0=16, d=2, c=3, k=16, roiSize=roiSize).cuda())
    dcfg = {"bsize": 4096, "nworker": 20, "collate": default_collate}
    # dcfg = {"bsize": 256, "nworker": 20, "collate": default_collate}

    model_name = "sp%d_rdn_small" % roiSize
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train_srdn_balance(roiSize=32):
    train_dataset = sdataset.SmallPatchDataset("train", roiSize, balance=True)
    val_dataset = sdataset.SmallPatchDataset("val", roiSize)
    test_dataset = sdataset.SmallPatchDataset("test", roiSize)

    from srdn import RDN
    # model = nn.DataParallel(
    #   RDN(g0=32, d=4, c=6, k=16, roiSize=roiSize).cuda())
    model = nn.DataParallel(RDN(g0=16, d=3, c=4, k=16, roiSize=roiSize).cuda())
    dcfg = {"bsize": 4096, "nworker": 20, "collate": default_collate}

    model_name = "sp%d_rdn_small_balance_d3c4k16" % roiSize
    train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg)


def train(model, model_name, train_dataset, val_dataset, test_dataset, dcfg):
    train_loader = DataLoader(train_dataset, batch_size=dcfg['bsize'],
                              shuffle=True, num_workers=dcfg['nworker'],
                              collate_fn=dcfg['collate'])

    model_dir = os.path.join("./modeldir/%s" % model_name)
    model_pth = os.path.join(model_dir, "model.pth")
    if os.path.exists(model_pth):
        print("----load pretrained model------")
        model = torch.load(model_pth)

    writer = tensorboardX.SummaryWriter(model_dir)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001, weight_decay=0.001)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=0.0001, weight_decay=0.001)
    criterion = torch.nn.BCELoss()

    epochs = 10000
    step = 1
    val_step = 1

    for e in range(epochs):
        model.train()
        st = time.time()

        for i_batch, sample_batched in enumerate(train_loader):
            (img, label) = sample_batched
            inputs = img.type(torch.cuda.FloatTensor)
            gt = label.type(torch.cuda.FloatTensor)
            model.zero_grad()
            pd = model(inputs)
            loss = criterion(pd, gt)
            loss.backward()
            optimizer.step()
            # for name, param in model.named_parameters():
            #     writer.add_histogram(
            #         name, param.clone().cpu().data.numpy(), step)
            writer.add_scalar("loss", loss, step)
            step += 1

        et = time.time()
        writer.add_scalar("train time", et - st, e)

        val_loss = run_val(model, val_dataset, writer,
                           val_step, criterion, dcfg)
        val_step += 1

        if e == 0:
            start_loss = val_loss
            min_loss = start_loss

        # if val_loss > 2 * min_loss:
        #     print("early stopping at %d" % e)
        #     break

        if e % 20 == 0:
            path = os.path.join(model_dir, "%d.pt" % e)
            torch.save(model.state_dict(), path)
            result = os.path.join(model_dir, "result_epoch%d.txt" % e)
            run_test(model, test_dataset, result, dcfg)

        if min_loss > val_loss:
            min_loss = val_loss
            print("----save best epoch:%d, loss:%f---" % (e, val_loss))
            torch.save(model, model_pth)
            result = os.path.join(model_dir, "result.txt")
            run_test(model, test_dataset, result, dcfg)


def run_val(model, val_dataset, writer, val_step, criterion, dcfg):
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(val_dataset, batch_size=dcfg['bsize'],
                                shuffle=False, num_workers=dcfg['nworker'],
                                collate_fn=dcfg['collate'])
        tot_loss = 0.0

        np_label = []
        np_pd = []
        for i_batch, sample_batched in enumerate(val_loader):
            (img, label) = sample_batched
            inputs = img.type(torch.cuda.FloatTensor)
            gt = label.type(torch.cuda.FloatTensor)
            pd = model(inputs)
            loss = criterion(pd, gt)
            tot_loss += loss

            val_pd = torch_util.threshold_tensor_batch(pd)
            np_pd.append(val_pd.data.cpu().numpy())
            np_label.append(gt.data.cpu().numpy())

        np_label = np.concatenate(np_label)
        np_pd = np.concatenate(np_pd)

        tot_loss = tot_loss / len(val_loader)
        writer.add_scalar("val loss", tot_loss.item(), val_step)
        torch_util.torch_metrics(np_label, np_pd, writer, val_step)

        return tot_loss.item()


def run_test(model, test_dataset, result, dcfg):
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=dcfg['bsize'],
                                 shuffle=False, num_workers=dcfg['nworker'],
                                 collate_fn=dcfg['collate'])
        np_label = []
        np_pd = []
        for i_batch, sample_batched in enumerate(test_loader):
            (img, label) = sample_batched
            inputs = img.type(torch.cuda.FloatTensor)
            gt = label.type(torch.cuda.FloatTensor)
            pd = model(inputs)
            test_pd = torch_util.threshold_tensor_batch(pd)
            np_pd.append(test_pd.data.cpu().numpy())
            np_label.append(gt.data.cpu().numpy())

        np_label = np.concatenate(np_label)
        np_pd = np.concatenate(np_pd)
        npmetrics.write_metrics(np_label, np_pd, result)


if __name__ == "__main__":
    # train_simg(depth=18)
    # train_sallpatch(depth=50, roiSize=100)
    # train_spatch(depth=50, roiSize=200)
    # train_sfpn()
    # train_sallpatch_sfpn()
    # train_sfpn_npatch()
    # train_sfpn_shuffle()
    # train_sfpn_pt()
    # train_sfpn_small()
    # train_sfpn_small_balance()
    # train_sfpn_small_aug()
    # train_srdn()
    train_srdn_balance()
