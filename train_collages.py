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
from sklearn.metrics import confusion_matrix
import math
from sklearn.metrics import classification_report
from efficientnet_pytorch import EfficientNet
from model.residual_attention_network import ResidualAttentionModel_92


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc
def per_class_accuracy(y_preds,y_true,class_labels):
    return [np.mean([
        (y_true[pred_idx] == np.round(y_pred)) for pred_idx, y_pred in enumerate(y_preds)
      if y_true[pred_idx] == int(class_label)
                    ]) for class_label in class_labels]

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

        self.model = torchvision.models.resnet50(pretrained=True)
        #num_ftrs = self.model.fc.in_features
        #self.model.fc = nn.Linear(num_ftrs, 4)
        n_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), #256
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 4))
            #nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.model(x)




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

config["batch_size"] = 16 #40, 8
config["batch_size_val"] = 16
#config["validation_batch_size"] = 1

config["crop_size"] = [224, 224]
config["res_size"] = [250, 400]#[512, 512]
config["n_channels"] = 1


config["lr"] = 3e-4
config["lr_decay"] = 0.7
config["patience"] = 3

config["epochs"] = 50  # number of epochs for training
config["workers"] = 5

config["crops"] = False # False # training on crops
config["patch_based"] = False # False #False  # False  # patch-based validation if config["crops"]=True
config["resize"] = True  # training with resize

config["thr"] = 0.5  # threshold for model`s output segmentation
config["mode"] = 'train'


config['path_to_exist_model'] =  None #'/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/saved_models/' \
                                #'damage_models/model_collages_all_resnet101_noWeight_moreAlb2.pt' #model_collages_all_effnetb6_noWeight_moreAlb_flips_250x400.pt'

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
    RandomBrightnessContrast,
    RandomGamma,
    CenterCrop,
    pytorch)


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

    #model =  MyResNet() #MVCNN() # MyVGG() # MyResNet()
    # model = ResidualAttentionModel_92()
    # model.fc =  nn.Linear(2048,4)
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=4)
    opt = torch.optim.Adam(model.parameters(), lr=0.00005)
    loss = nn.CrossEntropyLoss()

    if not os.path.exists(pathlib.Path().absolute().parent.parent / 'saved_models/damage_models/'):
        os.makedirs(pathlib.Path().absolute().parent.parent / 'saved_models/damage_models/')

    if not os.path.exists(pathlib.Path().absolute().parent.parent / 'saved_logs/damage_logs/'):
        os.makedirs(pathlib.Path().absolute().parent.parent / 'saved_logs/damage_logs/')

    model_path = Path((pathlib.Path().absolute().parent.parent / 'saved_models/damage_models/'
                       / args.model_name).with_suffix('.pt'))



    log = Path((pathlib.Path().absolute().parent.parent / 'saved_logs/damage_logs/'
                / args.model_name).with_suffix('.log')).open('at',  encoding='utf8')
    name = args.model_name + '_A-error'
    log1 = Path((pathlib.Path().absolute().parent.parent / 'saved_logs/damage_logs/'
                / name).with_suffix('.log')).open('at',  encoding='utf8')
    name = args.model_name + '_B-error'
    log2 = Path((pathlib.Path().absolute().parent.parent / 'saved_logs/damage_logs/'
                / name).with_suffix('.log')).open('at',  encoding='utf8')
    name = args.model_name + '_C-error'
    log3 = Path((pathlib.Path().absolute().parent.parent / 'saved_logs/damage_logs/'
                / name).with_suffix('.log')).open('at',  encoding='utf8')
    name = args.model_name +'_D-error'
    log4 = Path((pathlib.Path().absolute().parent.parent / 'saved_logs/damage_logs/'
                / name).with_suffix('.log')).open('at',  encoding='utf8')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_start = 0

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if config['path_to_exist_model'] is not None:
        model_path_exist = Path(config['path_to_exist_model'])
        if model_path_exist.exists():
            state = torch.load(str(model_path_exist))
            epoch_start = state['epoch']
            model.load_state_dict(state['model'])
            print('Restored model, epoch {}'.format(epoch_start))
        else:

            print('Train new model')

    model.to(device)

    X_train = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/x_train_NewCollagesAll.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'image_name' not in row[0]:
                X_train.append(row)
    y_train = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/y_train_NewCollagesAll.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'class' not in row[0]:
                y_train.append(row)

    X_val = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/x_val_NewCollagesAll.csv',
              'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'image_name' not in row[0]:
                X_val.append(row)
    y_val = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/y_val_NewCollagesAll.csv',
              'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'class' not in row[0]:
                y_val.append(row)
    X_test = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/x_test_NewCollagesAll.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'image_name' not in row[0]:
                X_test.append(row)
    y_test = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/y_test_NewCollagesAll.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'class' not in row[0]:
                y_test.append(row)

    # w_0 = 6256 / 4698
    # w_1 = 6256 / 6256
    # w_2 = 6256 / 3904
    # w_3 = 6256 / 456
    # # nSamples = [4698, 6256, 3904, 456]
    # # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    # # normedWeights = torch.FloatTensor(normedWeights).to(device)
    # normedWeights = torch.FloatTensor([w_0, w_1, w_2, w_3]).to(device)
    # print(normedWeights)
    # weights=torch.FloatTensor([0.2, 0.4, 0.2, 0.2]).to(device)
    loss = nn.CrossEntropyLoss()  #weight=normedWeights

    print("Number images for training={0}, number of images for validation={1}, "
          "number of images for test={2}".format(len(X_train), len(X_val), len(X_test)))

    def train_transform(p=1):
        return Compose([
            # Rotate(90, p=p),
            VerticalFlip(p=p),
            HorizontalFlip(p=p),
            RandomBrightnessContrast(p=p),
            RandomGamma(p=p),
            Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],p=1)
            # RandomCrop(32, 32)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225], p=1)
            # CenterCrop(512,515)
        ], p=p)


    train_loader = make_loader(X_train, y_train, dataset=ClassificationDataset_Collage, config=config,
                               shuffle=True, transforms=train_transform(p=0.5), batch_size=config["batch_size"])


    valid_loader = make_loader(X_val, y_val, dataset=ClassificationDataset_Collage, config=config,
                               shuffle=False, transforms=val_transform(p=1), batch_size=config["batch_size_val"])

    test_loader = make_loader(X_test, y_test, dataset=ClassificationDataset_Collage, config=config,
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
        train_accuracy = []
        train_acc12 = []
        train_acc13 = []
        train_acc14 = []
        train_acc21 = []
        train_acc23 = []
        train_acc24 = []
        train_acc31 = []
        train_acc32 = []
        train_acc34 = []
        train_acc41 = []
        train_acc42 = []
        train_acc43 = []

        val_acc12 = []
        val_acc13 = []
        val_acc14 = []
        val_acc21 = []
        val_acc23 = []
        val_acc24 = []
        val_acc31 = []
        val_acc32 = []
        val_acc34 = []
        val_acc41 = []
        val_acc42 = []
        val_acc43 = []


        test_acc12 = []
        test_acc13 = []
        test_acc14 = []
        test_acc21 = []
        test_acc23 = []
        test_acc24 = []
        test_acc31 = []
        test_acc32 = []
        test_acc34 = []
        test_acc41 = []
        test_acc42 = []
        test_acc43 = []



        train_acc1 = []
        train_acc2 =[]
        train_acc3 =[]
        train_acc4 =[]

        val_acc1 = []
        val_acc2 =[]
        val_acc3 =[]
        val_acc4 =[]


        test_acc1 = []
        test_acc2 =[]
        test_acc3 =[]
        test_acc4 =[]

        validation_acc = []
        test_acc = []
        validation_precision = []
        tq = tqdm.tqdm(total=(len(train_loader) *config["batch_size"]))
        tq.set_description('Training, Epoch {}'.format(epoch))
        count_train_step = 0
        try:
            model.train(True)
            random.seed()
            #exp_lr_scheduler.step()
            for (imgs, y_group, y_class) in train_loader:
                imgs = imgs.type('torch.FloatTensor').cuda()
                #print(imgs.shape)
                y_group = y_group.type('torch.LongTensor').cuda()
                y_class = y_class.type('torch.LongTensor').cuda()
                y = y_class
                batch_size = imgs.size(0)
                tq.update(batch_size)
                logits = model(imgs)


                _, preds = torch.max(logits, 1)

                # print(preds, masks, torch.sum(preds == masks))
                #print(logits , y_true)
                _loss = loss(logits, y)
                #print(logits, preds, y_true)
                acc = accuracy(logits, y) #binary_acc(logits, y_true) #
                matrix = confusion_matrix(y.cpu().detach().numpy(), preds.cpu().detach().numpy(), labels=[0,1,2,3],
                                          normalize='true')
                res = per_class_accuracy(y.cpu().detach().numpy(), preds.cpu().detach().numpy(), ['0', '1', '2', '3'])

                train_acc12.append(matrix[0, 1])
                train_acc13.append(matrix[0, 2])
                train_acc14.append(matrix[0, 3])

                train_acc21.append(matrix[1, 0])
                train_acc23.append(matrix[1, 2])
                train_acc24.append(matrix[1, 3])

                train_acc31.append(matrix[2, 0])
                train_acc32.append(matrix[2, 1])
                train_acc34.append(matrix[2, 3])

                train_acc41.append(matrix[3, 0])
                train_acc42.append(matrix[3, 1])
                train_acc43.append(matrix[3, 2])

                if not math.isnan(res[0]):
                    train_acc1.append(res[0])
                if not math.isnan(res[1]):
                    train_acc2.append(res[1])
                if not math.isnan(res[2]):
                    train_acc3.append(res[2])
                if not math.isnan(res[3]):
                    train_acc4.append(res[3])
                train_accuracy.append(acc)
                # train_dice.append(
                #     torch.sum(preds == masks).cpu().detach().numpy()/ len(preds))

                train_loss.append(_loss.item())

                tq.set_postfix(loss='{:.5f}'.format(np.mean(train_loss[-10:])), accuracy='{:.5f}'.format(np.mean(train_accuracy[-10:])))
                opt.zero_grad()
                _loss.backward()
                opt.step()

                count_train_step += 1
            tq.close()

            time_train += (time.time() - t1)

            #
            count_val_step = 0
            tq = tqdm.tqdm(total=(len(valid_loader)*config["batch_size_val"]))
            tq.set_description('Validation, Epoch {}'.format(epoch))
            model.train(False)
            with torch.no_grad():
                for (imgs, y_group, y_class) in valid_loader:
                    imgs = imgs.type('torch.FloatTensor').cuda()
                    y_group = y_group.type('torch.LongTensor').cuda()
                    y_class = y_class.type('torch.LongTensor').cuda()
                    y = y_class
                    batch_size = imgs.size(0)
                    tq.update(batch_size)

                    logits = model(imgs)

                    _, preds = torch.max(logits, 1)
                    matrix = confusion_matrix(y.cpu().detach().numpy(), preds.cpu().detach().numpy(),
                                              labels=[0, 1, 2, 3],
                                              normalize='true')
                    res = per_class_accuracy(y.cpu().detach().numpy(), preds.cpu().detach().numpy(),
                                             ['0', '1', '2', '3'])

                    val_acc12.append(matrix[0, 1])
                    val_acc13.append(matrix[0, 2])
                    val_acc14.append(matrix[0, 3])

                    val_acc21.append(matrix[1, 0])
                    val_acc23.append(matrix[1, 2])
                    val_acc24.append(matrix[1, 3])

                    val_acc31.append(matrix[2, 0])
                    val_acc32.append(matrix[2, 1])
                    val_acc34.append(matrix[2, 3])

                    val_acc41.append(matrix[3, 0])
                    val_acc42.append(matrix[3, 1])
                    val_acc43.append(matrix[3, 2])

                    if not math.isnan(res[0]):
                        val_acc1.append(res[0])
                    if not math.isnan(res[1]):
                        val_acc2.append(res[1])
                    if not math.isnan(res[2]):
                        val_acc3.append(res[2])
                    if not math.isnan(res[3]):
                        val_acc4.append(res[3])

                    acc = accuracy(logits, y)
                    validation_acc.append(acc)
                    validation_precision.append(precision_score(y.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro'))

                    tq.set_postfix(accuracy='{:.5f}'.format(np.mean(validation_acc[-10:])))

                    count_val_step += 1

                tq.close()
            scheduler.step(sum(validation_acc) / len(validation_acc))

            count_test_step = 0
            tq = tqdm.tqdm(total=(len(test_loader) * config["batch_size_val"]))
            tq.set_description('Testing, Epoch {}'.format(epoch))
            model.train(False)
            with torch.no_grad():
                for (imgs, y_group, y_class) in test_loader:
                    imgs = imgs.type('torch.FloatTensor').cuda()
                    y_group = y_group.type('torch.LongTensor').cuda()
                    y_class = y_class.type('torch.LongTensor').cuda()
                    y = y_class
                    batch_size = imgs.size(0)
                    tq.update(batch_size)

                    logits = model(imgs)

                    _, preds = torch.max(logits, 1)
                    matrix = confusion_matrix(y.cpu().detach().numpy(), preds.cpu().detach().numpy(),
                                              labels=[0, 1, 2, 3],
                                              normalize='true')
                    res = per_class_accuracy(y.cpu().detach().numpy(), preds.cpu().detach().numpy(),
                                             ['0', '1', '2', '3'])

                    test_acc12.append(matrix[0, 1])
                    test_acc13.append(matrix[0, 2])
                    test_acc14.append(matrix[0, 3])

                    test_acc21.append(matrix[1, 0])
                    test_acc23.append(matrix[1, 2])
                    test_acc24.append(matrix[1, 3])

                    test_acc31.append(matrix[2, 0])
                    test_acc32.append(matrix[2, 1])
                    test_acc34.append(matrix[2, 3])

                    test_acc41.append(matrix[3, 0])
                    test_acc42.append(matrix[3, 1])
                    test_acc43.append(matrix[3, 2])

                    if not math.isnan(res[0]):
                        test_acc1.append(res[0])
                    if not math.isnan(res[1]):
                        test_acc2.append(res[1])
                    if not math.isnan(res[2]):
                        test_acc3.append(res[2])
                    if not math.isnan(res[3]):
                        test_acc4.append(res[3])

                    acc = accuracy(logits, y)
                    test_acc.append(acc)

                    tq.set_postfix(accuracy='{:.5f}'.format(np.mean(test_acc[-10:])))

                    count_test_step += 1

                tq.close()
            write_event(log, epoch, loss_train =str(np.mean(train_loss)),  acc_train=str(np.mean(train_accuracy)),
                        acc_val=str(np.mean(validation_acc)), acc_test=str(np.mean(test_acc)), acc_valA=str(np.mean(val_acc1)),
                        acc_valB=str(np.mean(val_acc2)), acc_valC=str(np.mean(val_acc3)),
                        acc_valD=str(np.mean(val_acc4)), acc_testA=str(np.mean(test_acc1)),
                        acc_testB=str(np.mean(test_acc2)), acc_testC=str(np.mean(test_acc3)),
                        acc_testD=str(np.mean(test_acc4)))

            write_event(log1, epoch, error1=str(np.mean(val_acc12)), error2=str(np.mean(val_acc13)), error3=str(np.mean(val_acc14)),
                        error1t=str(np.mean(test_acc12)), error2t=str(np.mean(test_acc13)), error3t=str(np.mean(test_acc14)))
            write_event(log2, epoch, error1=str(np.mean(val_acc21)), error2=str(np.mean(val_acc23)), error3=str(np.mean(val_acc24)),
                        error1t=str(np.mean(test_acc21)), error2t=str(np.mean(test_acc23)), error3t=str(np.mean(test_acc24)))
            write_event(log3, epoch, error1=str(np.mean(val_acc31)), error2=str(np.mean(val_acc32)), error3=str(np.mean(val_acc34)),
                        error1t=str(np.mean(test_acc31)), error2t=str(np.mean(test_acc32)), error3t=str(np.mean(test_acc34)))
            write_event(log4, epoch, error1=str(np.mean(val_acc41)), error2=str(np.mean(val_acc42)), error3=str(np.mean(val_acc43)),
                        error1t=str(np.mean(test_acc41)), error2t=str(np.mean(test_acc42)), error3t=str(np.mean(test_acc43)))
            print("Train loss:", np.mean(train_loss))
            print("Train loss:", np.mean(train_loss))
            print("Final accuracy train:", np.mean(train_accuracy))
            print("Final accuracy validation:", np.mean(validation_acc))
            print("Final accuracy test:", np.mean(test_acc))

            print('Classes train', np.mean(train_acc1), np.mean(train_acc2), np.mean(train_acc3), np.mean(train_acc4))
            print('Classes val', np.mean(val_acc1), np.mean(val_acc2), np.mean(val_acc3), np.mean(val_acc4))
            print('Classes test', np.mean(test_acc1), np.mean(test_acc2), np.mean(test_acc3), np.mean(test_acc4))
            print('1 mistaken', np.mean(val_acc12), np.mean(val_acc13), np.mean(val_acc14))
            print('2 mistaken', np.mean(val_acc21), np.mean(val_acc23), np.mean(val_acc24))
            print('3 mistaken', np.mean(val_acc31), np.mean(val_acc32), np.mean(val_acc34))
            print('4 mistaken', np.mean(val_acc41), np.mean(val_acc42), np.mean(val_acc43))
            save(epoch)
            # print('Test:')
            # print('1 mistaken', np.mean(val_acc12), np.mean(val_acc13), np.mean(val_acc14))
            # print('2 mistaken', np.mean(val_acc21), np.mean(val_acc23), np.mean(val_acc24))
            # print('3 mistaken', np.mean(val_acc31), np.mean(val_acc32), np.mean(val_acc34))
            # print('4 mistaken', np.mean(val_acc41), np.mean(val_acc42), np.mean(val_acc43))
        except KeyboardInterrupt:
            tq.close()
            # print('Ctrl+C, saving snapshot')
            save(epoch)
            # print('done.')
            return
    print("final time train = {0}, final time val = {1}".format(time_train / 60, time_val / 60))


if __name__ == '__main__':
    main()
