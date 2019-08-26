#!/usr/bin/env python
# -*- coding: utf-8 -*-

# multi instance train model
import tensorboardX
import time
import os
import torch
import torch.nn as nn
import numpy as np
from util import torch_util
from util import npmetrics
from transformer import Transformer
from model import fvloader
from model import matloader

# BATCH_SIZE = 64
BATCH_SIZE = 32


def run_origin_train(model, dloader, imbtrain_data, writer, step, criterion):
    print("------run origin imblance train data-----------", step)
    model.eval()
    with torch.no_grad():
        st = time.time()

        for item in dloader.batch_fv(imbtrain_data, len(imbtrain_data)):
            genes, nimgs, labels, timesteps = item

            inputs = torch.from_numpy(nimgs).type(torch.cuda.FloatTensor)
            gt = torch.from_numpy(labels).type(torch.cuda.FloatTensor)
            pd = model(inputs)

            # loss = criterion(pd, gt)
            # criterion = torch.nn.BCELoss(reduction='none')
            all_loss = criterion(pd, gt)
            label_loss = torch.mean(all_loss, dim=0)
            loss = torch.mean(label_loss)

            for i in range(6):
                writer.add_scalar("origin sl_%d_loss" % i,
                                  label_loss[i].item(), step)
            writer.add_scalar("origin loss", loss.item(), step)

            val_pd = torch_util.threshold_tensor_batch(pd)
            # val_pd = torch.ge(pd, 0.5)
            np_pd = val_pd.data.cpu().numpy()
            torch_util.torch_metrics(labels, np_pd, writer, step,
                                     mode="origin")

        et = time.time()
        writer.add_scalar("origin time", et - st, step)
        return loss.item()


def run_val(model, dloader, val_data, writer, val_step, criterion):
    print("------run val-----------", val_step)
    model.eval()
    with torch.no_grad():
        st = time.time()

        for item in dloader.batch_fv(val_data, len(val_data)):
            genes, nimgs, labels, timesteps = item

            inputs = torch.from_numpy(nimgs).type(torch.cuda.FloatTensor)
            gt = torch.from_numpy(labels).type(torch.cuda.FloatTensor)
            pd = model(inputs)

            # loss = criterion(pd, gt)
            # criterion = torch.nn.BCELoss(reduction='none')
            all_loss = criterion(pd, gt)
            label_loss = torch.mean(all_loss, dim=0)
            loss = torch.mean(label_loss)

            for i in range(6):
                writer.add_scalar("val sl_%d_loss" % i,
                                  label_loss[i].item(), val_step)
            writer.add_scalar("val loss", loss.item(), val_step)

            val_pd = torch_util.threshold_tensor_batch(pd)
            # val_pd = torch.ge(pd, 0.5)
            np_pd = val_pd.data.cpu().numpy()
            lab_f1_macro = torch_util.torch_metrics(
                labels, np_pd, writer, val_step)

        et = time.time()
        writer.add_scalar("val time", et - st, val_step)
        return loss.item(), lab_f1_macro


def run_test(model, dloader, test_data, result):
    model.eval()
    with torch.no_grad():
        for item in dloader.batch_fv(test_data, len(test_data)):
            genes, nimgs, labels, timesteps = item

            inputs = torch.from_numpy(nimgs).type(torch.cuda.FloatTensor)
            pd = model(inputs)
            test_pd = torch_util.threshold_tensor_batch(pd)
            np_pd = test_pd.data.cpu().numpy()

            npmetrics.write_metrics(labels, np_pd, result)


def garbage_shuffle(train_data):
    genes, nimgs, labels, timesteps = zip(*train_data)
    np_labels = np.array(labels)
    np.random.shuffle(np_labels)
    s_labels = list(np_labels)
    garbage_data = list(zip(genes, nimgs, s_labels, timesteps))
    return garbage_data


def train(fv, model_name, criterion, size=0):
    if fv == "matlab":
        dloader = matloader
    else:
        dloader = fvloader

    train_data = dloader.load_train_data(size=size, balance=False)
    val_data = dloader.load_val_data(size=size)
    test_data = dloader.load_test_data(size=size)
    # model_name = "transformer_%s_size%d_bce" % (fv, size)
    model_dir = os.path.join("./modeldir/%s" % model_name)
    model_pth = os.path.join(model_dir, "model.pth")

    writer = tensorboardX.SummaryWriter(model_dir)

    if os.path.exists(model_pth):
        print("------load model--------")
        model = torch.load(model_pth)
    else:
        model = Transformer(fv, NUM_HEADS=4, NUM_LAYERS=3).cuda()
        # model = Transformer(fv).cuda()
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0001, weight_decay=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer,
    #         patience=100)

    epochs = 1000
    step = 1
    val_step = 1
    max_f1 = 0.0

    for e in range(epochs):
        model.train()
        print("------epoch--------", e)
        st = time.time()

        train_shuffle = fvloader.shuffle(train_data)
        for item in fvloader.batch_fv(train_shuffle, batch=BATCH_SIZE):

            # for name, param in model.named_parameters():
            #     writer.add_histogram(
            #         name, param.clone().cpu().data.numpy(), step)

            # writer.add_histogram(
            #     "grad/"+name, param.grad.clone().cpu().data.numpy(), step)
            model.zero_grad()

            genes, nimgs, labels, timesteps = item
            inputs = torch.from_numpy(nimgs).type(torch.cuda.FloatTensor)

            gt = torch.from_numpy(labels).type(torch.cuda.FloatTensor)
            pd = model(inputs)

            # loss = criterion(pd, gt)
            all_loss = criterion(pd, gt)
            label_loss = torch.mean(all_loss, dim=0)
            loss = torch.mean(label_loss)
            # for i in range(6):
            #     writer.add_scalar("train sl_%d_loss" % i,
            #                       label_loss[i].item(), step)

            train_pd = torch_util.threshold_tensor_batch(pd)
            np_pd = train_pd.data.cpu().numpy()
            torch_util.torch_metrics(
                labels, np_pd, writer, step, mode="train")

            writer.add_scalar("train loss", loss, step)
            loss.backward()
            optimizer.step()
            step += 1

        et = time.time()
        writer.add_scalar("train time", et - st, e)
        # for param_group in optimizer.param_groups:
        #     writer.add_scalar("lr", param_group['lr'], e)

        # run_origin_train(model, imbtrain_data, writer, e, criterion)

        if e % 1 == 0:
            val_loss, val_f1 = run_val(
                model, dloader, val_data, writer, val_step, criterion)
            # scheduler.step(val_loss)
            val_step += 1
            if e == 0:
                start_loss = val_loss
                min_loss = start_loss

            # if val_loss > 2 * min_loss:
            #     print("early stopping at %d" % e)
            #     break
            # if e % 50 == 0:
            #     pt = os.path.join(model_dir, "%d.pt" % e)
            #     torch.save(model.state_dict(), pt)
            #     result = os.path.join(model_dir, "result_epoch%d.txt" % e)
            #     run_test(model, test_data, result)

            if min_loss > val_loss or max_f1 < val_f1:
                if min_loss > val_loss:
                    print("---------save best----------", "loss", val_loss)
                    min_loss = val_loss
                if max_f1 < val_f1:
                    print("---------save best----------", "f1", val_f1)
                    max_f1 = val_f1
                torch.save(model, model_pth)
                result = os.path.join(model_dir, "result_epoch%d.txt" % e)
                run_test(model, dloader, test_data, result)


def final_test(fv="res18-128", size=2):
    test_data = fvloader.load_test_data(size=size)

    model_name = "transformer_%s_size%d" % (fv, size)
    model_dir = os.path.join("./modeldir/%s" % model_name)
    model_pth = os.path.join(model_dir, "model.pth")

    if os.path.exists(model_pth):
        print("------load model for test--------")
        model = torch.load(model_pth)
    else:
        raise Exception("train model first")

    model.eval()
    result = os.path.join(model_dir, "result.txt")
    run_test(model, test_data, result)


if __name__ == "__main__":
    # train("res18-128", 0)
    train("matlab", 0)
