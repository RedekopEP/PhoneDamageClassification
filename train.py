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

        self.model = torchvision.models.resnet34(pretrained=True) #torchvision.models.googlenet(pretrained=True) #torchvision.models.mobilenet_v2(num_classes=4) #resnet34(pretrained=True)

        #self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)



class MyVGG(nn.Module):

    def __init__(self, in_channels=18):
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


class MVCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(MVCNN, self).__init__()
        model = models.resnet34(pretrained=True)

        fc_in_features = model.fc.in_features

        # model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fc_in_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, inputs):  # inputs.shape = samples x views x channels x height x width
        inputs = inputs.transpose(0, 1)
        #print(inputs.shape)
        view_features = []
        for view_batch in inputs:
            view_batch = self.features(view_batch)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            view_features.append(view_batch)

        pooled_views, _ = torch.max(torch.stack(view_features), 0)
        outputs = self.classifier(pooled_views)
        return outputs

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

config = dict()
config["model"] = "UNet"   # choices = ["UNet", "FPN", "PSPNet", "Linknet"]
config["loss"] = "DiceBCE"   # choices = ["DiceBCE", "FocalTversky", "Lovasz", "Focal", "Tversky"]
config["optimizer"] = "Adam"  # choices = ["Adam", "SGD"]

config["dice_weight"] = 1 # if config["loss"] = "DiceBCE"

config["batch_size"] = 5
config["batch_size_val"] = 5
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


config['path_to_exist_model'] = None# config['path_to_exist_model'] = '/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/saved_models/damage_models/model_400_againstD.pt'  #None #'/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/saved_models/damage_models/model_400_class_cut.pt'  # provide path to model, if you want to continue its training
#ne_sem_seg/stage1_experiment/saved_models/damage_models/overfit_model_400.pt'  # provide path to model, if you want to continue its training

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
    model = MyVGG() #MVCNN() # MyVGG() # MyResNet()
    # opt = torch.optim.Adam(model.classifier.parameters(), lr=0.0005)
    opt = torch.optim.Adam(model.parameters(), lr=0.00005)
    #opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    #opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

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
            state = torch.load(str(model_path_exist))
            epoch_start = state['epoch']
            model.load_state_dict(state['model'])
            print('Restored model, epoch {}'.format(epoch_start))
        else:

            print('Train new model')

    model.to(device)

    X_train = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/x_train_cut_400.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'image_name' not in row[0]:
                X_train.append(row)
    y_train = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/y_train_cut_400.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'class' not in row[0]:
                y_train.append(row)
    X_test = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/x_test_cut_400.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'image_name' not in row[0]:
                X_test.append(row)
    y_test = []
    with open('/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/y_test_cut_400.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'class' not in row[0]:
                y_test.append(row)
    # nSamples = [240, 120]
    # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    # normedWeights = torch.FloatTensor(normedWeights).to(device)
    # print(normedWeights)
    #weights=torch.FloatTensor([0.2, 0.4, 0.2, 0.2]).to(device)
    loss = nn.CrossEntropyLoss()
    # X_test = X_test[:500]
    # y_test = y_test[:500]

    print("Number images for training={0}, number of images for validation={1}".format(len(X_train),
                                                                                       len(X_test)))

    def train_transform(p=1):
        return Compose([
            Rotate(90, p=p),
            VerticalFlip(p=p),
            HorizontalFlip(p=p)#,
            # Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225],p=p)
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

    train_loader = make_loader(X_train, y_train, dataset=ClassificationDataset_Pie, config=config,
                               shuffle=True, transforms=train_transform(p=1), batch_size=config["batch_size"])


    # train_loader = make_loader(train_file_names, train_file_names_masks, dataset=PhonesDataset, config=config,
    #                           shuffle=True,  transforms=train_transform(p=1), batch_size=config["batch_size"])

    valid_loader = make_loader(X_test, y_test, dataset=ClassificationDataset_Pie, config=config,
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
        train_acc1 = []
        train_acc2 =[]
        train_acc3 =[]
        train_acc4 =[]

        val_acc1 = []
        val_acc2 =[]
        val_acc3 =[]
        val_acc4 =[]


        validation_acc = []
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
                # matrix = confusion_matrix(y.cpu().detach().numpy(), preds.cpu().detach().numpy())
                # res = matrix.diagonal() / matrix.sum(axis=1)
                res = per_class_accuracy(y.cpu().detach().numpy(), preds.cpu().detach().numpy(), ['0', '1'])
                #print(y.cpu().detach().numpy().shape, preds.cpu().detach().numpy().shape, res.shape)
                if not math.isnan(res[0]):
                    train_acc1.append(res[0])
                if not math.isnan(res[1]):
                    train_acc2.append(res[1])
                # if not math.isnan(res[2]):
                #     train_acc3.append(res[2])
                # if not math.isnan(res[3]):
                #     train_acc4.append(res[3])
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
                    y_group= y_group.type('torch.LongTensor').cuda()
                    y_class= y_class.type('torch.LongTensor').cuda()
                    y = y_class
                    batch_size = imgs.size(0)
                    tq.update(batch_size)

                    logits = model(imgs)

                    _, preds = torch.max(logits, 1)
                    # matrix = confusion_matrix(y.cpu().detach().numpy(), preds.cpu().detach().numpy())
                    # res = matrix.diagonal() / matrix.sum(axis=1)
                    res = per_class_accuracy(y.cpu().detach().numpy(), preds.cpu().detach().numpy(),
                                             ['0', '1'])
                    if not math.isnan(res[0]):
                        val_acc1.append(res[0])
                    if not math.isnan(res[1]):
                        val_acc2.append(res[1])
                    # if not math.isnan(res[2]):
                    #     val_acc3.append(res[2])
                    # if not math.isnan(res[3]):
                    #     val_acc4.append(res[3])
                    # print(precision_score(y.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro'),
                    #       np.unique(y.cpu().detach().numpy()),
                    #       np.unique(preds.cpu().detach().numpy()))
                    #print(torch.sigmoid(logits).cpu(), np.round(torch.sigmoid(logits).cpu()), y_true)

                    #print(metrics.confusion_matrix(y_true.cpu().detach().numpy(), preds.cpu().detach().numpy()))

                    #print(preds, masks, torch.sum(preds == masks))
                    #
                    # validation_dice.append(
                    #     torch.sum(preds == masks).cpu().detach().numpy() / len(preds))
                    acc = accuracy(logits, y) #binary_acc(logits, y_true) #accuracy(logits, y_true)
                    validation_acc.append(acc)
                    validation_precision.append(precision_score(y.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro'))

                    tq.set_postfix(accuracy='{:.5f}'.format(np.mean(validation_acc[-10:])))

                    count_val_step += 1

                tq.close()
            scheduler.step(sum(validation_acc) / len(validation_acc))
            write_event(log, epoch, loss_train =str(np.mean(train_loss)),  acc_train=str(np.mean(train_accuracy)), acc_val=str(np.mean(validation_acc)))
            print("Train loss:", np.mean(train_loss))
            print("Final accuracy train:", np.mean(train_accuracy))
            print("Final accuracy validation:", np.mean(validation_acc))
            print("Final precision validation:", np.mean(validation_precision))

            print('Classes train', np.mean(train_acc1), np.mean(train_acc2), np.mean(train_acc3), np.mean(train_acc4))
            print('Classes val', np.mean(val_acc1), np.mean(val_acc2), np.mean(val_acc3), np.mean(val_acc4))
        except KeyboardInterrupt:
            tq.close()
            # print('Ctrl+C, saving snapshot')
            save(epoch)
            # print('done.')
            return
    print("final time train = {0}, final time val = {1}".format(time_train / 60, time_val / 60))


if __name__ == '__main__':
    main()
