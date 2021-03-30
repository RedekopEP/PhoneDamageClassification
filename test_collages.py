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
        num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, 4)
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

config["batch_size"] = 40 #8
config["batch_size_val"] = 40
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


config['path_to_exist_model'] = '/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/saved_models/' \
                                'damage_models/model_collages_all_resnet50L_noWeight_moreAlb_flips_250x400.pt'

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

    model = MyResNet()


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

    print("number of images for test={0}".format( len(X_test)))

    def val_transform(p=1):
        return Compose([
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225], p=1)
            # CenterCrop(512,515)
        ], p=p)

    test_loader = make_loader(X_test, y_test, dataset=ClassificationDataset_Collage, config=config,
                               shuffle=False, transforms=val_transform(p=1), batch_size=config["batch_size_val"])

    # # UNFREEZE ALL THE WEIGHTS OF THE NETWORK
    # for param in model.parameters():
    #     param.requires_grad = True

    time_train = 0
    time_val = 0
    for epoch in range(1):


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



        test_acc1 = []
        test_acc2 =[]
        test_acc3 =[]
        test_acc4 =[]

        validation_acc = []
        test_acc = []

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
                print(y.cpu().detach().numpy(), preds.cpu().detach().numpy())
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
            print("Final accuracy test:", np.mean(test_acc))
            print('Classes test', np.mean(test_acc1), np.mean(test_acc2), np.mean(test_acc3), np.mean(test_acc4))

    print("final time train = {0}, final time val = {1}".format(time_train / 60, time_val / 60))


if __name__ == '__main__':
    main()
