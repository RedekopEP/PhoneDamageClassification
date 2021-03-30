import json
import tqdm
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import torch.backends.cudnn as cudnn
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn
import os
import cv2
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ClassificationDataset, ClassificationDataset_Pie, ClassificationDataset_Collage
from init import init
from glob import glob
import pathlib
import argparse
from patchify import patchify, unpatchify
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import csv
import pandas as pd
from ResNet_Att import Net
from torch.optim import lr_scheduler
import torch_optimizer as optim
from sklearn import metrics
from sklearn.metrics import precision_score


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc.cpu().detach().numpy()


class MyResNet(nn.Module):

    def __init__(self, in_channels=3):
        super(MyResNet, self).__init__()

        self.model = models.resnet34(pretrained=True) #models.mobilenet_v2(num_classes=4) #torchvision.models.googlenet(pretrained=True) #torchvision.models.mobilenet_v2(num_classes=4) #resnet34(pretrained=True)

        #self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #
        n_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)) # nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.model(x)


class Flatten(nn.Module):
    def forward(self, input):
        print('flatten', input.view(input.size(0), -1).shape)
        return input.view(input.size(0), -1)
    # def __init__(self):
    #     super.__init__(Flatten, self)


class B_CNN(nn.Module):

    def __init__(self):
        super(B_CNN, self).__init__()
        self.block1 = nn.Sequential()

        self.block1.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3))  # convolution
        self.block1.add_module('ReLU-1', nn.ReLU())
        self.block1.add_module('conv12', nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3))  # convolution
        self.block1.add_module('BatchNorm-12', nn.BatchNorm2d(64))
        self.block1.add_module('ReLU-1', nn.ReLU())
        self.block1.add_module('pool1', nn.MaxPool2d(2))  # max pooling 2x2

        self.block1.add_module('conv2', nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3))  # convolution
        self.block1.add_module('ReLU-2', nn.ReLU())
        self.block1.add_module('conv22',
                         nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3))  # convolution
        self.block1.add_module('BatchNorm-22', nn.BatchNorm2d(128))
        self.block1.add_module('ReLU-22', nn.ReLU())
        self.block1.add_module('pool2', nn.MaxPool2d(2))  # max pooling 2x2
        # --- coarse 1 branch ---

        self.branch1 = nn.Sequential()
        #self.branch1.add_module('flatten1', Flatten())
        self.branch1.add_module('dence3', nn.Linear(128*56*56, 256))
        self.branch1.add_module('ReLU-63', nn.ReLU())
        self.branch1.add_module('drop1', nn.Dropout(0.5))
        self.branch1.add_module('dence4', nn.Linear(256, 256))
        self.branch1.add_module('ReLU-64', nn.ReLU())
        self.branch1.add_module('drop2', nn.Dropout(0.5))
        self.branch1.add_module('dence5', nn.Linear(256, 2))
        self.branch1.add_module('sftm1', nn.Softmax())

        self.block2 = nn.Sequential()
        self.block2.add_module('conv3', nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3))  # convolution
        # model.add_module('BatchNorm-3', nn.BatchNorm2d(256))
        self.block2.add_module('ReLU-3', nn.ReLU())
        self.block2.add_module('conv31',
                         nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3))  # convolution
        # model.add_module('BatchNorm-31', nn.BatchNorm2d(256))
        self.block2.add_module('ReLU-31', nn.ReLU())
        self.block2.add_module('conv32',
                         nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3))  # convolution
        self.block2.add_module('BatchNorm-32', nn.BatchNorm2d(256))
        self.block2.add_module('ReLU-32', nn.ReLU())
        self.block2.add_module('pool3', nn.MaxPool2d(2))

        self.branch2 = nn.Sequential()
        self.branch2.add_module('flatten2', Flatten())
        self.branch2.add_module('dence6', nn.Linear(10, 1024))
        self.branch2.add_module('ReLU-73', nn.ReLU())
        self.branch2.add_module('drop3', nn.Dropout(0.5))
        self.branch2.add_module('dence7', nn.Linear(1024, 1024))
        self.branch2.add_module('ReLU-73', nn.ReLU())
        self.branch2.add_module('drop4', nn.Dropout(0.5))
        self.branch1 .add_module('dence8', nn.Linear(1024, 4))
        self.branch1 .add_module('sftm2', nn.Softmax())


    def forward(self, x):
        x = self.block1(x)
        print('1', x.shape)
        x_fl = x.reshape(x.size(0), -1)
        print('2', x_fl.shape)
        branch1 = self.branch1(x_fl)
        x = self.block2(x)
        print('3', x.shape)
        branch2 = self.branch2(x)
        return branch1, branch2



class MyVGG(nn.Module):

    def __init__(self, in_channels=1):
        super( MyVGG, self).__init__()
        self.model = models.vgg16(pretrained=False, num_classes=4)  # pretrained=False just for debug reasons
        first_conv_layer = [nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(self.model.features))
        self.model.features = nn.Sequential(*first_conv_layer)

    def forward(self, x):
        return self.model(x)


my_resnet = MyResNet()


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

config = dict()
config["model"] = "UNet"   # choices = ["UNet", "FPN", "PSPNet", "Linknet"]
config["loss"] = "DiceBCE"   # choices = ["DiceBCE", "FocalTversky", "Lovasz", "Focal", "Tversky"]
config["optimizer"] = "Adam"  # choices = ["Adam", "SGD"]

config["dice_weight"] = 1 # if config["loss"] = "DiceBCE"

config["batch_size"] = 4
config["batch_size_val"] = 4
#config["validation_batch_size"] = 1

config["crop_size"] = [224, 224]
config["res_size"] = [512, 512]#[512, 512]
config["n_channels"] = 1


config["lr"] = 3e-4
config["lr_decay"] = 0.7
config["patience"] = 3

config["epochs"] = 100  # number of epochs for training
config["workers"] = 5

config["crops"] = False #False # training on crops
config["patch_based"] = False #False #False  # False  # patch-based validation if config["crops"]=True
config["resize"] = True  # training with resize

config["thr"] = 0.5  # threshold for model`s output segmentation
config["mode"] = 'train'

config['path_to_exist_model'] = None  # provide path to model, if you want to continue its training

config["with_masks"] = True


def parse(parser):
    arg = parser.add_argument
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--dataset_name', default='')
    arg('--model_name', default='')
    args = parser.parse_args()
    return args


def make_loader(file_names, csv_filename, dataset, config=None, sampler=None, transforms=None, batch_size=1, shuffle=True):
    return DataLoader(
        dataset=dataset(file_names, csv_filename, config=config, transforms=transforms),
        num_workers=config["workers"],
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        sampler=sampler,
        shuffle=shuffle
    )



def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    Rotate,
Normalize,
    RandomCrop,
CenterCrop
)


class FocalLoss(nn.Module):

    def __init__(self, weight=None,
                 gamma=5, reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        ).mean()

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc) * 100
    return acc

def main():
    # ---- fix seeds and deterministic flags ----
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # ---------------------------------------

    args = parse(argparse.ArgumentParser())

    # ---- load model ----
    # model, loss = init(config)
    model = MyResNet() #B_CNN() # MyVGG() # MyResNet()
    # opt = torch.optim.Adam(model.classifier.parameters(), lr=0.0005)
    #opt = torch.optim.Adam(model.parameters(), lr=0.00005)
    #opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    if not os.path.exists(pathlib.Path().absolute().parent.parent / 'saved_models/damage_models/'):
        os.makedirs(pathlib.Path().absolute().parent.parent / 'saved_models/damage_models/')

    if not os.path.exists(pathlib.Path().absolute().parent.parent / 'saved_logs/damage_logs/'):
        os.makedirs(pathlib.Path().absolute().parent.parent / 'saved_logs/damage_logs/')

    model_path = Path((pathlib.Path().absolute().parent.parent / 'saved_models/damage_models/'
                       / args.model_name).with_suffix('.pt'))
    log = Path((pathlib.Path().absolute().parent.parent / 'saved_logs/damage_logs/'
                / args.model_name).with_suffix('.log')).open('at',  encoding='utf8')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_start = 0

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if config['path_to_exist_model'] is not None:
        model_path_exist = Path(config['path_to_exist_model'])
        if model_path_exist.exists():
            state = torch.load(str(model_path_exist ))
            epoch_start = state['epoch']
            model.load_state_dict(state['model'])
            print('Restored model, epoch {}'.format(epoch_start))
        else:

            print('Train new model')

    model.to(device)

    X_train = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/x_train_collage_cut.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'image_name' not in row[0]:
                X_train.append(row)
    y_train = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/y_train_collage_cut.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'class' not in row[0]:
                y_train.append(row)
    X_test = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/x_test_collage_cut.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'image_name' not in row[0]:
                X_test.append(row)
    y_test = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/y_test_collage_cut.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'class' not in row[0]:
                y_test.append(row)
    # nSamples = [240, 120]
    # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    # normedWeights = torch.FloatTensor(normedWeights).to(device)
    # print(normedWeights)
    weights=[0.2, 0.3, 0.2, 0.3]
    loss = nn.CrossEntropyLoss()

    print("Number images for training={0}, number of images for validation={1}".format(len(X_train),
                                                                                       len(X_test)))

    def train_transform(p=1):
        return Compose([
            Rotate(90, p=p),
            VerticalFlip(p=p),
            HorizontalFlip(p=p),
            Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],p=p)
            # RandomCrop(32, 32)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            # CenterCrop(512,515)
        ], p=p)

    # weights = np.zeros(len(train_file_names_masks), dtype=np.float32)
    # full_vol = 0
    # print('start_cycle')
    # for idx in range(len(train_file_names_masks)):
    #     mask = cv2.imread(train_file_names_masks[idx])
    #     mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    #     mask = np.array(mask > 0).astype(np.uint8)
    #
    #     num = np.count_nonzero(np.array(mask > 0).astype(np.uint8))
    #
    #     full_vol += num
    #     weights[idx] = num
    # print('finish_cycle')
    # weights = weights / full_vol
    #
    # weights = torch.DoubleTensor(weights)
    # sampler_weight = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    # count = 0
    # for y in y_test:
    #     if y == ['C']:
    #         count += 1
    # print(count)

    train_loader = make_loader(X_train, y_train, dataset=ClassificationDataset_Collage, config=config,
                               shuffle=True, transforms=train_transform(p=1), batch_size=config["batch_size"])


    # train_loader = make_loader(train_file_names, train_file_names_masks, dataset=PhonesDataset, config=config,
    #                           shuffle=True,  transforms=train_transform(p=1), batch_size=config["batch_size"])

    valid_loader = make_loader(X_test, y_test, dataset=ClassificationDataset_Collage, config=config,
                               shuffle=False, transforms=val_transform(p=1), batch_size=config["batch_size_val"])

    # ---------------------------------------
    exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)
    scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=2, mode='max', verbose=True)

    # UNFREEZE ALL THE WEIGHTS OF THE NETWORK
    for param in model.parameters():
        param.requires_grad = True

    time_train = 0
    time_val = 0
    for epoch in range(epoch_start, config["epochs"]):
        t1 = time.time()
        save = lambda ep: torch.save({
            'model':  model.state_dict(),
            'epoch': ep,
        }, str(model_path))

        batch_size = config["batch_size"]
        train_loss = []
        valid_loss =[]
        train_accuracy_group = []
        train_accuracy_class = []
        validation_acc = []
        validation_precision = []
        tq = tqdm.tqdm(total=(len(train_loader) *config["batch_size"]))
        tq.set_description('Training, Epoch {}'.format(epoch))
        count_train_step = 0
        try:
            model.train(True)
            random.seed()
            #exp_lr_scheduler.step()
            for (imgs,  y_group, y_class) in train_loader:
                imgs = imgs.type('torch.FloatTensor').cuda()
                y_class = y_class.type('torch.LongTensor').cuda()
                y_group = y_group.type('torch.LongTensor').cuda()
                y_class = y_group
                batch_size = imgs.size(0)
                tq.update(batch_size)
                #print('ings', imgs.shape)
                logits_class = model(imgs)

                _loss = loss(logits_class, y_class)

                # acc = accuracy(logits_group, y_group) #binary_acc(logits, y_true) #
                # train_accuracy_group.append(acc)
                acc = accuracy(logits_class, y_class)
                #print(logits_class, y_class)
                train_accuracy_class.append(acc)
                # train_dice.append(
                #     torch.sum(preds == masks).cpu().detach().numpy()/ len(preds))

                train_loss.append(_loss.item())

                tq.set_postfix(loss='{:.5f}'.format(np.mean(train_loss[-10:])), accuracy_group='{:.5f}'.format(np.mean(train_accuracy_group[-10:])),
                                                                                                         accuracy_class='{:.5f}'.format(np.mean(train_accuracy_class[-10:])))
                opt.zero_grad()
                _loss.backward()
                opt.step()

                count_train_step += 1
            tq.close()

            time_train += (time.time() - t1)

            #
            # count_val_step = 0
            # tq = tqdm.tqdm(total=(len(valid_loader)*config["batch_size_val"]))
            # tq.set_description('Validation, Epoch {}'.format(epoch))
            # model.train(False)
            # with torch.no_grad():
            #     for (imgs, y_group) in valid_loader:
            #         imgs = imgs.type('torch.FloatTensor').cuda()
            #         y_group = y_group.type('torch.LongTensor').cuda()
            #         batch_size = imgs.size(0)
            #         tq.update(batch_size)
            #
            #         logits = model(imgs)
            #
            #         _, preds = torch.max(logits, 1)
            #         print(precision_score(y_group.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro'),
            #               np.unique(y_group.cpu().detach().numpy()),
            #               np.unique(preds.cpu().detach().numpy()))
            #         #print(torch.sigmoid(logits).cpu(), np.round(torch.sigmoid(logits).cpu()), y_true)
            #
            #         #print(metrics.confusion_matrix(y_true.cpu().detach().numpy(), preds.cpu().detach().numpy()))
            #
            #         #print(preds, masks, torch.sum(preds == masks))
            #         #
            #         # validation_dice.append(
            #         #     torch.sum(preds == masks).cpu().detach().numpy() / len(preds))
            #         acc = accuracy(logits, y_group) #binary_acc(logits, y_true) #accuracy(logits, y_true)
            #         validation_acc.append(acc)
            #         validation_precision.append(precision_score(y_group.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro'))
            #
            #         tq.set_postfix(accuracy='{:.5f}'.format(np.mean(validation_acc[-10:])))
            #
            #         count_val_step += 1
            #
            #     tq.close()
            # scheduler.step(sum(validation_acc) / len(validation_acc))
            # print("Train loss:", np.mean(train_loss))
            # print("Final accuracy train:", np.mean(train_accuracy))
            # print("Final accuracy validation:", np.mean(validation_acc))
            # print("Final precision validation:", np.mean(validation_precision))
            print("Train loss:", np.mean(train_loss))
            #print("Final accuracy train group:", np.mean(train_accuracy_group))
            print("Final accuracy train class:", np.mean(train_accuracy_class))
        except KeyboardInterrupt:
            tq.close()
            # print('Ctrl+C, saving snapshot')
            save(epoch)
            # print('done.')
            return
    print("final time train = {0}, final time val = {1}".format(time_train / 60, time_val / 60))


if __name__ == '__main__':
    main()
