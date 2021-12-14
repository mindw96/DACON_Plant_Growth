import copy
import time
from glob import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class KistDataset(Dataset):
    def __init__(self, combination_df, is_test= None):
        self.combination_df = combination_df
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        self.is_test = is_test

    def __getitem__(self, idx):
        before_image = Image.open(self.combination_df.iloc[idx]['l_path'])
        after_image = Image.open(self.combination_df.iloc[idx]['r_path'])

        before_image = self.transform(before_image)
        after_image = self.transform(after_image)
        if self.is_test:
            return before_image, after_image
        time_delta = self.combination_df.iloc[idx]['time_delta']
        return before_image, after_image, time_delta

    def __len__(self):
        return len(self.combination_df)


class CustomDataset(Dataset):
    def __init__(self, path=None, is_test= None):
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        self.is_test = is_test

    def __getitem__(self, idx):
        length = len(self.path)
        before_rdn = random.randint(0, length - 1)
        after_rdn = random.randint(0, length - 1)

        before_day = int(self.path[before_rdn].split('.')[1].split('_')[2][3:])
        after_day = int(self.path[after_rdn].split('.')[1].split('_')[2][3:])

        while True:
            if before_day == after_day:
                after_rdn = random.randint(0, length - 1)
                after_day = int(self.path[after_rdn].split('.')[1].split('_')[2][3:])
            else:
                break

        before_image = Image.open(self.path[before_rdn])
        after_image = Image.open(self.path[after_rdn])

        before_image = self.transform(before_image)
        after_image = self.transform(after_image)
        if self.is_test:
            return before_image, after_image
        time_delta = before_day - after_day

        return before_image, after_image, time_delta

    def __len__(self):
        return len(self.path)


class LoadTrainData:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.path = './data/combine/*'
        self.file_list = glob.glob(self.path)
        self.file_list_png = [file for file in self.file_list if file.endswith('.png')]

        self.train_path, self.valid_path = train_test_split(self.file_list_png, test_size=0.2, shuffle=True, random_state=9608)

    def train_load(self):
        train_dataset = CustomDataset(self.train_path, is_test=False)
        train_data_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)

        return train_data_loader

    def valid_load(self):
        valid_dataset= CustomDataset(self.valid_path, is_test=False)
        valid_data_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size)

        return valid_data_loader


class LoadTestData:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.test_set = pd.read_csv('./data/test_dataset/test_data.csv')
        self.test_set['l_root'] = self.test_set['before_file_path'].map(
            lambda x: './data/test_dataset/' + x.split('_')[1] + '/' + x.split('_')[2])
        self.test_set['r_root'] = self.test_set['after_file_path'].map(
            lambda x: './data/test_dataset/' + x.split('_')[1] + '/' + x.split('_')[2])
        self.test_set['l_path'] = self.test_set['l_root'] + '/' + self.test_set['before_file_path'] + '.png'
        self.test_set['r_path'] = self.test_set['r_root'] + '/' + self.test_set['after_file_path'] + '.png'

    def test_load(self):
        test_dataset = KistDataset(self.test_set, is_test=True)
        test_data_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size)

        return test_data_loader

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


def train(model=None, epochs=0, train_loader=None, valid_loader=None, optimizer=None, criterion=None):
    best_loss = 100.0
    train_loss, valid_loss = [], []

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    since = time.time()

    for epoch in range(epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        train_batch_loss, valid_batch_loss = 0.0, 0.0
        model.train()
        batch_index = 0

        for before, after, delta in train_loader:
            before = before.to(device)
            after = after.to(device)
            delta = delta.unsqueeze(1)
            delta = delta.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(before, after)
                output = output.type(torch.FloatTensor)
                delta = delta.type(torch.FloatTensor)
                loss = criterion(output, delta)

            loss.backward()

            optimizer.step()

            train_batch_loss += loss.item()

            batch_index += 1

        train_loss.append(train_batch_loss / batch_index)

        model.eval()

        batch_index = 0

        for before, after, delta in valid_loader:
            before = before.to(device)
            after = after.to(device)
            delta = delta.unsqueeze(1)
            delta = delta.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(before, after)
                output = output.type(torch.FloatTensor)
                delta = delta.type(torch.FloatTensor)
                loss = criterion(output, delta)

            valid_batch_loss += loss.item()

            batch_index += 1

        valid_loss.append(valid_batch_loss / batch_index)

        print('Train Loss: {:.4f} Valid Loss: {:.4f}'.format(train_loss[epoch], valid_loss[epoch]))

        if valid_loss[epoch] < best_loss:
            best_idx = epoch
            best_loss = valid_loss[epoch]
            torch.save(model.state_dict(), 'check.pt')
            best_model_wts = copy.deepcopy(model.state_dict())
            print('==> best model saved - {} / {:.4f}'.format(best_idx + 1, best_loss))

    time_elapsed = time.time() - since
    print()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Loss: %d - %.4f' % (best_idx + 1, best_loss))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'Final.pt')
    print('final model saved')

    return model, train_loss, valid_loss, best_idx


def draw(train_loss, valid_loss, best_idx):
    print('best model : %d - %.4f' % (best_idx, valid_loss[best_idx]))
    fig, ax1 = plt.subplots()

    ax1.plot(train_loss, 'g-')
    ax1.plot(valid_loss, 'k-')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    ax1.set_ylabel('loss', color='k')
    ax1.tick_params('y', colors='k')

    fig.tight_layout()
    plt.show()


def test(model, test_loader):
    test_value = []
    with torch.no_grad():
        for test_before, test_after in test_loader:
            logit = model(test_before, test_after)
            value = logit.squeeze(1).detach().float()

            test_value.extend(value)

    return [int(x.numpy()) for x in test_value]


def submit(predict):
    sub_csv = pd.read_csv('data/sample_submission.csv')
    sub_csv['time_delta'] = predict
    sub_csv.to_csv('submit.csv', header=True, index=False)
