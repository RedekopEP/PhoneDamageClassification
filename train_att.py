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
from dataset import ClassificationDataset
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

resnet = torchvision.models.resnet18()


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc


class MyResNet(nn.Module):

    def __init__(self, in_channels=12):
        super(MyResNet, self).__init__()

        # bring resnet
        self.model = torchvision.models.resnet50()

        # original definition of the first layer on the renset class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # your case
        #self.model.features[0] = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.model.classifier[0].out_features = 4
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        # n_inputs = self.model.fc.in_features
        # self.model.fc = nn.Sequential(
        #     nn.Linear(n_inputs, 256), #256
        #     nn.ReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(256, 4),
        #     nn.LogSoftmax(dim=1))

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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = dict()
config["model"] = "UNet"   # choices = ["UNet", "FPN", "PSPNet", "Linknet"]
config["loss"] = "DiceBCE"   # choices = ["DiceBCE", "FocalTversky", "Lovasz", "Focal", "Tversky"]
config["optimizer"] = "Adam"  # choices = ["Adam", "SGD"]

config["dice_weight"] = 1 # if config["loss"] = "DiceBCE"

config["batch_size"] =5
config["batch_size_val"] = 5
#config["validation_batch_size"] = 1

config["crop_size"] = [224, 224]
config["res_size"] = [512, 512]
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


def calc_md(prediction, ground_truth):
    eps = 1e-15
    intersection = np.logical_and(prediction > 0, ground_truth > 0).astype(np.float32).sum()
    return (2. * intersection.sum() + eps) / ((prediction > 0).sum() + (ground_truth > 0).sum() + eps)

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
    RandomCrop,
CenterCrop
)


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
    model = Net(NumClasses=4)  #MyResNet()
    model.AddAttententionLayer()
    loss = nn.NLLLoss() #nn.CrossEntropyLoss()

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

    # ---------------------------------------
    #opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # ---- load data ----
    # img_names = glob("/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/datasets/nsys_photoset_10-12-20/*/*/*/")
    # img_names_new = []
    # for im in img_names:
    #     list_names = glob(im+ '*')
    #     if len(list_names) == 6:
    #         img_names_new.append(im)
    # img_names_new = img_names_new[:2000]
    # train_size = int(len(img_names_new) * 0.7)   # Number of training images:number of validation images = 80:20
    #
    # train_file_names = img_names_new[len(img_names_new) - train_size:]
    # val_file_names = img_names_new[:len(img_names_new) - train_size]
    X_train = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/x_train.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X_train.append(row)
    y_train = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/y_train.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            y_train.append(row)
    X_test = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/x_test.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X_test.append(row)
    y_test = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/y_test.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            y_test.append(row)

    X_train = X_train[:500]
    y_train = y_train[:500]

    X_test = X_test[:200]
    y_test = y_test[:200]

    print("Number images for training={0}, number of images for validation={1}".format(len(X_train),
                                                                                       len(X_test)))

    def train_transform(p=1):
        return Compose([
            Rotate(90, p=p),
            VerticalFlip(p=p),
            HorizontalFlip(p=p)#,
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

    train_loader = make_loader(X_train, y_train, dataset=ClassificationDataset, config=config,
                               shuffle=True, transforms=train_transform(p=1), batch_size=config["batch_size"])


    # train_loader = make_loader(train_file_names, train_file_names_masks, dataset=PhonesDataset, config=config,
    #                           shuffle=True,  transforms=train_transform(p=1), batch_size=config["batch_size"])

    valid_loader = make_loader(X_test, y_test,  dataset=ClassificationDataset, config=config,
                               shuffle=True, transforms=val_transform(p=1), batch_size=config["batch_size_val"])

    # ---------------------------------------

    scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=3, mode='max', verbose=True)

    best_acc = 0
    time_train = 0
    time_val = 0
    AVGLoss = 0
    for epoch in range(epoch_start, config["epochs"]):
        t1 = time.time()
        save = lambda ep: torch.save({
            'model':  model.state_dict(),
            'epoch': ep,
        }, str(model_path))

        batch_size = config["batch_size"]
        train_loss = []
        train_dice = []
        validation_dice = []
        tq = tqdm.tqdm(total=(len(train_loader) *config["batch_size"]))
        tq.set_description('Training, Epoch {}'.format(epoch))
        count_train_step = 0
        try:
            model.train(True)
            random.seed()

            for (imgs, rois, masks, LabelsOneHot) in train_loader:
                imgs = imgs.type('torch.FloatTensor').to(device)#.cuda()
                rois = rois.type('torch.FloatTensor').to(device)#.cuda()
                masks = masks.type('torch.LongTensor').to(device)#.cuda()
                LabelsOneHot =  LabelsOneHot.type('torch.LongTensor').to(device)#.cuda()
                # print(torch.unique(masks))
                # print(imgs.dtype)
                batch_size = imgs.size(0)
                tq.update(batch_size)

                Prob, Lb = model(imgs, rois)
                model.zero_grad()
                OneHotLabels = LabelsOneHot #torch.autograd.Variable(torch.from_numpy(LabelsOneHot).cuda(), requires_grad=False)
                Loss = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate cross entropy loss
                if AVGLoss == 0:
                    AVGLoss = float(Loss.data.cpu().numpy())  # Caclculate average loss for display
                else:
                    AVGLoss = AVGLoss * 0.999 + 0.001 * float(Loss.data.cpu().numpy())

                train_dice.append(
                    accuracy(torch.log(Prob + 0.0000001), masks).cpu().detach().numpy())
                Loss.backward()  # Backpropogate loss
                opt.step()  # Apply gradient decend change weight
                torch.cuda.empty_cache()

                # logits = model(imgs, rois)
                #
                # _loss = loss(logits, masks)
                #
                # train_dice.append(
                #     accuracy(logits, masks).cpu().detach().numpy())
                #
                # # print(_loss.item(), multi_acc(logits, masks))
                #
                # train_loss.append(_loss.item())
                #
                # tq.set_postfix(dice='{:.5f}'.format(np.mean(train_dice[-10:])))
                # opt.zero_grad()
                # _loss.backward()
                # opt.step()
                #
                # count_train_step += 1
            tq.close()

            time_train += (time.time() - t1)

            #
            count_val_step = 0
            tq = tqdm.tqdm(total=(len(valid_loader)*config["batch_size_val"]))
            tq.set_description('Validation, Epoch {}'.format(epoch))
            model.train(False)
            for (imgs, rois, masks, LabelsOneHot) in valid_loader:
                imgs = imgs.type('torch.FloatTensor').to(device)  # .cuda()
                rois = rois.type('torch.FloatTensor').to(device)  # .cuda()
                masks = masks.type('torch.LongTensor').to(device)  # .cuda()
                LabelsOneHot = LabelsOneHot.type('torch.LongTensor').to(device)  # .cuda()
                # print(torch.unique(masks))
                # print(imgs.dtype)
                batch_size = imgs.size(0)
                tq.update(batch_size)

                Prob, Lb = model(imgs, rois)
                model.zero_grad()
                validation_dice.append(
                    accuracy(torch.log(Prob + 0.0000001), masks).cpu().detach().numpy())


                tq.set_postfix(dice='{:.5f}'.format(np.mean(validation_dice[-10:])))

                count_val_step += 1
            tq.close()
            print("Train loss:", np.mean(train_loss))
            print("Final accuracy train:", np.mean(train_dice))
            print("Final accuracy validation:", np.mean(validation_dice))


        except KeyboardInterrupt:
            tq.close()
            # print('Ctrl+C, saving snapshot')
            # save(epoch)
            # print('done.')
            return
    print("final time train = {0}, final time val = {1}".format(time_train / 60, time_val / 60))


if __name__ == '__main__':
    main()
