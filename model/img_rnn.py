#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as utils_rnn
import os
import time
# import fvloader
import matloader
import tensorboardX
from util import npmetrics
from util import torch_util


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

HIDDEN_SIZE = 128
NUM_CLASSES = 6


class ImageRNN(nn.Module):

    def __init__(self, input_size, hidden_size, nclasses, bidirectional=False):
        super(ImageRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nclasses = nclasses
        self.bid = bidirectional

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=self.bid)
        self.fc = nn.Linear(hidden_size, nclasses)

    def forward(self, s_nimgs, s_timesteps, hidden=None):
        s_nimgs = torch.transpose(s_nimgs, 0, 1)
        s_nimgs_p = utils_rnn.pack_padded_sequence(s_nimgs, s_timesteps)

        output_pack, hidden = self.gru(s_nimgs_p, hidden)
        output, output_len = utils_rnn.pad_packed_sequence(output_pack)

        # print("out", output.shape)
        # print("hidden", hidden.shape)
        if self.bid:
            last_hidden = hidden[0, :, :] + hidden[1, :, :]
        else:
            last_hidden = hidden.squeeze()
        out = self.fc(last_hidden)
        return torch.sigmoid(out), hidden


def run_val(rnn, val_data, writer, val_step, criterion):
    print("------run val-----------", val_step)
    rnn.eval()
    with torch.no_grad():
        st = time.time()

        # for item in fvloader.batch_fv(val_data, len(val_data)):
        for item in matloader.batch_fv(val_data, len(val_data)):
            hidden = None
            genes, nimgs, labels, timesteps = item
            idx = np.argsort(np.array(-timesteps))
            s_nimgs = torch.from_numpy(
                np.stack(nimgs[idx])).type(torch.cuda.FloatTensor)
            s_labels = torch.from_numpy(
                labels[idx]).type(torch.cuda.FloatTensor)
            s_timesteps = timesteps[idx]
            out_pack, hidden = rnn(s_nimgs, s_timesteps)

        loss = criterion(out_pack, s_labels)
        writer.add_scalar("val loss", loss.item(), val_step)

        val_pd = torch_util.threshold_tensor_batch(out_pack)
        np_pd = val_pd.data.cpu().numpy()
        lab_f1_macro = torch_util.torch_metrics(
            labels[idx], np_pd, writer, val_step)

        et = time.time()
        writer.add_scalar("val time", et - st, val_step)
        return loss.item(), lab_f1_macro


def run_test(rnn, test_data, result):
    rnn.eval()
    with torch.no_grad():
        # for item in fvloader.batch_fv(test_data, len(test_data)):
        for item in matloader.batch_fv(test_data, len(test_data)):
            genes, nimgs, labels, timesteps = item
            idx = np.argsort(np.array(-timesteps))

            s_nimgs = torch.from_numpy(
                np.stack(nimgs[idx])).type(torch.cuda.FloatTensor)
            s_timesteps = timesteps[idx]
            out_pack, hidden = rnn(s_nimgs, s_timesteps)

        test_pd = torch_util.threshold_tensor_batch(out_pack)
        npmetrics.write_metrics(labels[idx], np.array(test_pd), result)


def train(fv="res18-128", size=0):
    # train_data = fvloader.load_train_data(size=size, balance=True)
    # val_data = fvloader.load_val_data(size=size)
    # test_data = fvloader.load_test_data(size=size)
    train_data = matloader.load_train_data(size=size, balance=True)
    val_data = matloader.load_val_data(size=size)
    test_data = matloader.load_test_data(size=size)

    model_name = "imgrnn_%s_size%d_bce_gbalance" % (fv, size)
    model_dir = os.path.join("./modeldir/%s" % model_name)
    model_pth = os.path.join(model_dir, "model.pth")

    # INPUT_DIM = int(fv.split("-")[-1])
    INPUT_DIM = 1097

    writer = tensorboardX.SummaryWriter(model_dir)

    if os.path.exists(model_pth):
        print("------load model--------")
        rnn = torch.load(model_pth)
    else:
        rnn = ImageRNN(INPUT_DIM, HIDDEN_SIZE, NUM_CLASSES).cuda()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.00001)
    criterion = torch.nn.BCELoss(reduce=True, size_average=True)

    epochs = 10000
    step = 1
    val_step = 1
    max_f1 = 0.0
    for e in range(epochs):
        print("------epoch--------", e)
        rnn.train()
        st = time.time()

        # train_shuffle = fvloader.shuffle(train_data)
        # for item in fvloader.batch_fv(train_shuffle, batch=256):
        train_shuffle = matloader.shuffle(train_data)
        for item in matloader.batch_fv(train_shuffle, batch=64):

            rnn.zero_grad()

            genes, nimgs, labels, timesteps = item
            idx = np.argsort(np.array(-timesteps))

            # s_genes = genes[idx]
            s_nimgs = torch.from_numpy(
                np.stack(nimgs[idx])).type(torch.cuda.FloatTensor)
            s_labels = torch.from_numpy(
                labels[idx]).type(torch.cuda.FloatTensor)
            s_timesteps = timesteps[idx]
            # print("timesteps", timesteps, len(timesteps))
            # print("sorted timesteps", s_timesteps)
            # print("s_nimgs", s_nimgs.shape, s_nimgs[0].shape)
            # print("s_labels", s_labels.shape)
            out, hidden = rnn(s_nimgs, s_timesteps)
            # print("pd", out.shape)
            loss = criterion(out, s_labels)

            writer.add_scalar("loss", loss, step)
            loss.backward()
            optimizer.step()
            step += 1

        et = time.time()
        writer.add_scalar("train time", et - st, e)

        if e % 1 == 0:
            val_loss, val_f1 = run_val(
                rnn, val_data, writer, val_step, criterion)
            val_step += 1
            if e == 0:
                start_loss = val_loss
                min_loss = start_loss

            # if val_loss > 2 * min_loss:
            #     print("early stopping at %d" % e)
            #     break

            # if e % 50 == 0:
            #     pt = os.path.join(model_dir, "%d.pt" % e)
            #     torch.save(rnn.state_dict(), pt)
            #     result = os.path.join(model_dir, "result_epoch%d.txt" % e)
            #     run_test(rnn, test_data, result)

            if min_loss > val_loss or max_f1 < val_f1:
                if min_loss > val_loss:
                    print("---------save best----------", "loss", val_loss)
                    min_loss = val_loss
                if max_f1 < val_f1:
                    print("---------save best----------", "f1", val_f1)
                    max_f1 = val_f1
                torch.save(rnn, model_pth)
                result = os.path.join(model_dir, "result_epoch%d.txt" % e)
                run_test(rnn, test_data, result)


if __name__ == "__main__":
    # train()
    train(fv='matlab', size=0)
