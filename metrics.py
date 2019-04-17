import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


class Metric():
    def __init__(self, train_loader, val_loader, test_loader, model, optimizer, loss, bins,
                 test_prev_out_continuous, test_out_continuous, max_val):
        self.bins = bins
        self.test_prev_out_continuous = test_prev_out_continuous
        self.test_out_continuous = test_out_continuous
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.bins = bins
        self.optimizer = optimizer
        self.loss = loss
        self.max_val = max_val

        
    def train(self):
        total = 0.0
        total_loss = 0.0
        for iter, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs.requires_grad_()

            self.model.init_hidden(mode='train')

            self.model.zero_grad()
            output = self.model(inputs)


            loss = self.loss(output, labels)


            loss.backward()
            clipping_value = 1  # arbitrary number of your choosing
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping_value)
            self.optimizer.step()

            total += 1
            total_loss += loss.item()
            loss_val = total_loss / total

        return loss_val

    def val_eval(self):
        total = 0.0
        total_loss = 0.0
        for iter, data in enumerate(self.val_loader):
            inputs, labels = data
            inputs.requires_grad_()

            self.model.init_hidden(mode='val')

            # for evaluating the network, we disable the gradient calculation with the no_grad function
            with torch.no_grad():
                output = self.model(inputs)

            loss = self.loss(output, labels)

            total += 1
            total_loss += loss.item()
            loss_val = total_loss / total
        return loss_val

    def test_eval(self, graph=False):
        for iter, data in enumerate(self.test_loader):
            inputs, labels = data
        self.model.init_hidden(mode='test')
        # for evaluating the network, we disable the gradient calculation with the no_grad function
        with torch.no_grad():
            output = self.model(inputs)
        percent = 1 + self.bins[output.max(1)[1].numpy()]

        prev_out_continuous = self.test_prev_out_continuous[0:len(percent)]
        target_out_continuous = self.test_out_continuous[0:len(percent)]

        pred_out_continuous = np.multiply(percent, prev_out_continuous)
        if graph:
            for idx in range(len(pred_out_continuous) - 201):
                plt.plot(np.arange(0, 200), pred_out_continuous[idx:idx + 200] * self.max_val,
                         target_out_continuous[idx:idx + 200] * self.max_val)
                plt.legend(['prediction', 'target'])
                plt.draw()
                plt.pause(.001)
                plt.clf()
        else:
            loss_val = mean_absolute_error(pred_out_continuous, target_out_continuous) * self.max_val
            mae_std = np.std(abs(pred_out_continuous - target_out_continuous)) * self.max_val

        return loss_val, mae_std

    def graph(self):
        self.test_eval(graph=True)