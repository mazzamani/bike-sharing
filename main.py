import torch
from torch import nn
from sklearn.metrics import mean_absolute_error
from dataset_loader import BikeDataset
from torch.utils.data import DataLoader
from models import GRUmodel
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import save_results


def update_params(loaded_data, mode):
    total = 0.0
    total_loss = 0.0
    for iter, data in enumerate(loaded_data):
        inputs, labels = data
        inputs.requires_grad_()
        if mode == 'train':
            model.train()
            model.zero_grad()
        else:
            model.eval()

        model.init_hidden(mode=mode)
        output = model(inputs)

        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(output, labels)

        if mode == 'train':
            loss.backward()
            clipping_value = 1  # arbitrary number of your choosing
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
        total += 1
        total_loss += loss.item()
        loss_val = total_loss / total
        mae_std = 0.
        if mode == 'test' or mode == 'graph':
            percent = 1 + training_set.bins[output.max(1)[1].numpy()]

            prev_out_continuous = test_set.prev_out_continuous[0:len(percent)]
            target_out_continuous = test_set.out_continuous[0:len(percent)]

            pred_out_continuous = np.multiply(percent, prev_out_continuous)
            loss_val = mean_absolute_error(pred_out_continuous, target_out_continuous) * model.max_val
            mae_std = np.std(abs(pred_out_continuous - target_out_continuous)) * model.max_val

            if mode == 'graph':
                for idx in range(len(pred_out_continuous) - 201):
                    plt.plot(np.arange(0, 200), pred_out_continuous[idx:idx + 200] * model.max_val,
                             target_out_continuous[idx:idx + 200] * model.max_val)
                    plt.legend(['prediction', 'target'])
                    plt.draw()
                    plt.pause(.001)
                    plt.clf()

    return loss_val, mae_std


def update(mode):
    '''
    :param mode: can be train, val, test, graph. Except the train mode, the network is in evaluation mode
    :return: loss value
    '''
    # TODO without torch eval and merge update and update_param function
    if mode == 'train':
        loaded_data = train_loader
        evaluate = False

    elif mode == 'val':
        loaded_data = val_loader
        evaluate = True

    elif mode == 'test' or mode == 'graph':
        loaded_data = test_loader
        evaluate = True

    # for evaluating the network, we disable the gradient calculation with the no_grad function
    if evaluate:
        with torch.no_grad():
            loss = update_params(loaded_data, mode)
    else:
        loss = update_params(loaded_data, mode)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hourly Prediction of the Bike Dataset using PyTorch')

    # experiment setup
    parser.add_argument('-rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('-batch_size', default=32, type=int, help='minibatch size')
    parser.add_argument('-epochs', default=200, type=int, help='train iters each timestep')

    # model
    parser.add_argument('-model', type=str, default='gru', help='only GRU is used for this file')
    parser.add_argument('-hidden_dim', type=int, default=5, help='hidden dimension in RNN hidden layer')
    parser.add_argument('-num_layers', type=int, default=1, help='the number of RNN  layers')
    parser.add_argument('-dropout', type=float, default=0.0, help='the probability for dropout [default: 0.0]')

    # Dataset
    parser.add_argument('-seqlen', type=int, default=2,
                        help='number of output classes in the generated training dataset')
    parser.add_argument('-prev_cnt', dest='prev_cnt', type=str, default='hour',
                        help='Model(x_t, y_(t-i), y_(t-j): values: no, hour, week')
    parser.add_argument('-day_num', dest='day_num', type=int, default=1,
                        help='Model(x_t, y_(t-i), y_(t-j): values: no, hour, week')
    parser.add_argument('--reduced', dest='reduced', action='store_false',
                        help='if true, only uses the limited number of features which filtered by simple L1 regression')
    parser.add_argument('--bidir', dest='bidir', action='store_true',
                        help='if true, the lstm model will use the last hidden state as output otherwise average over '
                             'hidden states')
    parser.set_defaults(reduced=False)
    parser.set_defaults(bidir=False)
    args = parser.parse_args()

    seq_len = args.seqlen

    # percent_bins = [-1 + i / 10 for i in range(31)] #+ [1.2 + i / 5 for i in range(10)]
    percent_bins = [-1] + list(np.arange(-.75, .45, .05)) + list(np.arange(.5, 2.75, .25))
    percent_bins.sort()

    # loading csv file
    dataset = pd.read_csv("dataset/hour.csv")
    training_set = BikeDataset(set_type='train', dataset=dataset, seq_len=seq_len, prev_cnt=args.prev_cnt,
                               reduced_features=args.reduced, percet_bins=percent_bins, day_num=args.day_num)

    val_set = BikeDataset(set_type='val', dataset=dataset, seq_len=seq_len, prev_cnt=args.prev_cnt,
                          day_num=args.day_num,
                          reduced_features=args.reduced, percet_bins=percent_bins, max_cnt=training_set.max_cnt,
                          repeated_data_num=training_set.repeated_data_num)

    test_set = BikeDataset(set_type='test', dataset=dataset, seq_len=seq_len, prev_cnt=args.prev_cnt,
                           day_num=args.day_num,
                           reduced_features=args.reduced, percet_bins=percent_bins, max_cnt=training_set.max_cnt,
                           repeated_data_num=val_set.repeated_data_num)

    train_loader = DataLoader(training_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set),
                            shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=len(test_set),
                             shuffle=False, drop_last=True)

    feature_size = training_set.feature_length
    if args.prev_cnt == 'no':
        previous_cnt_feature_size = 0
    elif args.prev_cnt == 'hour':
        previous_cnt_feature_size = 1
    elif args.prev_cnt == 'day':
        previous_cnt_feature_size = args.day_num

    model = GRUmodel(args, input_dim=feature_size + previous_cnt_feature_size, bi_dir=args.bidir,
                     val_test_batch=[len(val_set), len(test_set)], class_num=len(percent_bins))

    # get the maximum value from the training as a reference to convert back normalize data to the orginal number
    model.max_val = training_set.max_cnt.numpy()
    optimizer = optim.Adam(model.parameters(), lr=args.rate)

    loss_function = nn.CrossEntropyLoss

    train_loss_records = []
    val_loss_records = []
    test_loss_records = []

    val_std_records = []
    test_std_records = []
    subfolder_settled = False
    subfolder = ''
    filename = ''
    results_file = ''

    mistake_counter = 0  # mistakes counter for validation loss
    for epoch in range(args.epochs):
        train_loss, _ = update('train')
        train_loss_records.append(train_loss)

        val_loss, val_std = update('val')
        val_loss_records.append(val_loss)
        val_std_records.append(val_std)

        if epoch > 20:
            if val_loss_records[-1] > val_loss_records[-2]:
                mistake_counter += 1

        test_loss, test_std = update('test')
        test_loss_records.append(test_loss)
        test_std_records.append(test_std)

        print(
            '[Epoch: %3d/%3d] Train CELoss: %.4f,    Val CELoss: %.4f,   Test MAE: %.4f, STD: %.4f'
            % (epoch, args.epochs, train_loss_records[epoch], val_loss_records[epoch], test_loss_records[epoch],
               test_std_records[epoch]))
        if mistake_counter > 30 or epoch == args.epochs - 1:
            print('TRAINING TERMINATED:30 time in a row validation loss has increased')
            subfolder_settled, subfolder, filename, results_file = save_results(subfolder_settled, subfolder, filename,
                                                                                results_file, args=args, model=model,
                                                                                test_loss=test_loss, test_std=test_std,
                                                                                epoch=epoch)
            update('graph')
            break
