import numpy as np
from scipy.ndimage import distance_transform_edt as distance
import torch
from torch.utils.data import Dataset
import scipy
import cv2
import scipy.ndimage
from skimage import filters
from patchify import patchify, unpatchify
import csv
from glob import glob
import os
import torch_optimizer as optim
from PIL import Image, ImageOps


def z_score_normalization(img):
    img2 = img - img.mean(axis=(0, 1), keepdims=True)
    img2 = img2 / (img.std(axis=(0, 1), keepdims=True) + 1e-16)
    return img2.astype(np.float32)


def min_max_normalization(img):
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-14)
    return img_norm


def transform_to_square(_img, _size):
    """
    Resizes PIL image to (_size X _size) with saving aspect ratio.
    :param _img: PIL image.
    """
    SIZE = _size
    #print('HERE', _img.shape)
    # max boarder - to required size with saving aspect ratio
    before = _img.size
    before_p = np.unique(np.array(_img))
    _img.thumbnail((SIZE, SIZE), Image.ANTIALIAS)

    # adding black boarders
    delta_w = SIZE - _img.size[0]
    delta_h = SIZE - _img.size[1]
    pad_left, pad_top, pad_right, pad_bot = 0, 0, 0, 0
    if delta_h:
        pad_top = delta_h // 2
        pad_bot = delta_h - pad_top
    if delta_w:
        pad_left = delta_w // 2
        pad_right = delta_w - pad_left
    _img = np.asarray(ImageOps.expand(_img, (pad_left, pad_top, pad_right, pad_bot)))

    return _img


class ClassificationDataset(Dataset):

    def __init__(self, X, y, config, transforms):
        self.file_names = X
        self.config = config
        self.transforms = transforms
        self.classes = y

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_names = self.file_names[idx][0]
        image, phone_mask, buttons_mask, damage_mask = load_image(img_file_names)


        class_ = self.classes[idx][0]

        y_oneHot = np.zeros(4)
        if class_ == 'A':
            y_ = 0
            y_c = 0
            y_oneHot[0] = 1
        elif class_ == 'B':
            y_ = 0
            y_c = 0
        elif class_ == 'C':
            y_ = 1
            y_c = 0
        elif class_ == 'D':
            y_ = 1
            y_c = 1

        Img = []
        Dam = []

        if image.shape[0]!=6:
            print("!!!! wow",  img_file_names)
        for i in range(image.shape[0]):
            phone_mask_ = cv2.erode(phone_mask[i], np.ones((5, 5)), iterations=1)
            #x,y,z = np.where(phone_mask_)
            image_ = image[i]
            dam_ = damage_mask[i]
            if image_.shape[0] == 4032:
                image_= cv2.resize(image_, (756, 1008))
            if phone_mask_.shape[0] == 4032:
                phone_mask_= cv2.resize(phone_mask_, (756, 1008))
            if dam_ .shape[0] == 4032:
                dam_ = cv2.resize(dam_ , (756, 1008))
            #print(np.unique(buttons_mask[i]))
            image_ = image_*phone_mask_ #image_[min(x):max(x), min(y):max(y)] #image_*phone_mask_
            dam_ = dam_ * phone_mask_ * (1-buttons_mask[i]/255).astype(np.uint8)
            dam_ = cv2.cvtColor(dam_, cv2.COLOR_RGB2GRAY)
            # image_ = cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY)
            # image_ = filters.sobel(image_).astype(np.float32)
            # image_ = cv2.cvtColor(image_, cv2.COLOR_GRAY2RGB)
            # sh1 = image_.shape[0]
            # sh2 = image_.shape[1]
            # image_before = image_
            # image_ = Image.fromarray(image_, 'RGB')
            # image_ = transform_to_square(image_, 512)

            image_ = cv2.resize(image_, (self.config["res_size"][0], self.config["res_size"][1]))
            dam_ = cv2.resize(dam_ , (self.config["res_size"][0], self.config["res_size"][1]))

            data = {"image": image_, "roi": dam_}

            augmented = self.transforms(**data)
            image_ = augmented["image"]
            roi = augmented["roi"]

            #image_ = np.dstack((image_, dam_[:, :, None]))#np.array([image_, dam_])
            Dam.append(roi)
            image_ = min_max_normalization(image_)
            Img.append(image_)

        roi = np.array(Dam)
        image = np.array(Img)
        #print(image.shape)


        if len(image.shape) != 4:
            image = image[:, :,:, None]
            image = image.transpose((0, 3,1, 2))
            roi = roi[:, :, :, None]
            roi = roi.transpose((0, 3, 1, 2))
        #
        else:
            image = image.transpose((0, 3, 1, 2))
            #roi = roi.transpose((0, 3, 1, 2))
        #print(roi.shape)
        return torch.from_numpy(image).to(torch.float32), torch.tensor(y_).to(torch.float32), torch.tensor(y_c).to(torch.float32) #torch.from_numpy(roi).to(torch.float32), , torch.from_numpy(y_oneHot).to(torch.float32)


class ClassificationDataset_Pie(Dataset):

    def __init__(self, X, y, config, transforms):
        self.file_names = X
        self.config = config
        self.transforms = transforms
        self.classes = y

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_names = self.file_names[idx][0]
        image, phone_mask, buttons_mask, damage_mask = load_image(img_file_names)

        class_ = self.classes[idx][0]

        y_oneHot = np.zeros(4)
        if class_ == 'A':
            y_ = 0
            y_c = 0
            y_oneHot[0] = 1
        elif class_ == 'B':
            y_ = 0
            y_c = 1
        elif class_ == 'C':
            y_ = 1
            y_c = 2
        elif class_ == 'D':
            y_ = 1
            y_c = 3

        Img = []
        Dam = []
        # if image.shape[0]!=6:
        #     print("!!!! wow", img_file_names, image.shape[0])
        for i in range(image.shape[0]):
            phone_mask_ = cv2.erode(phone_mask[i], np.ones((5, 5)), iterations=1)
            # x,y,z = np.where(phone_mask_)
            image_ = image[i]
            dam_ = damage_mask[i]
            if image_.shape[0] == 4032:
                image_ = cv2.resize(image_, (756, 1008))
            if phone_mask_.shape[0] == 4032:
                phone_mask_ = cv2.resize(phone_mask_, (756, 1008))
            # print(np.unique(buttons_mask[i]))
            image_ = image_ * phone_mask_  # image_[min(x):max(x), min(y):max(y)] #image_*phone_mask_
            dam_ = dam_ * phone_mask_ * (1 - buttons_mask[i] / 255).astype(np.uint8)
            dam_ = cv2.cvtColor(dam_, cv2.COLOR_RGB2GRAY)
            image_ = cv2.resize(image_, (self.config["res_size"][0], self.config["res_size"][1]))
            Dam.append(damage_mask[i])
            #Dam = cv2.resize(damage_mask[i], (self.config["res_size"][0], self.config["res_size"][1]))
            #print(image_.shape)
            image_ = min_max_normalization(image_)
            Img.append(image_)
        damage = np.dstack((Dam[0], Dam[1], Dam[2], Dam[3],  Dam[4],  Dam[5]))
        image = np.dstack((Img[0], Img[1],  Img[2], Img[3],  Img[4], Img[5])) #,
        #print(image.shape)
        #image = Img[0]
        #damage = Dam[0]

        # image = z_score_normalization(image)

        data = {"image": image, "roi":damage}

        augmented = self.transforms(**data)
        image = augmented["image"]
        roi = augmented["roi"]

        if len(roi.shape) != 3:
            roi = roi[:, :, None]
        image = image.transpose((2, 0, 1))
        roi = roi.transpose((2, 0, 1))
        #print(image.shape)
        return torch.from_numpy(image).to(torch.float32), torch.tensor(y_).to(torch.float32), torch.tensor(y_c).to(
            torch.float32)  # t


class ClassificationDataset_Collage(Dataset):

    def __init__(self, X, y, config, transforms):
        self.file_names = X
        self.config = config
        self.transforms = transforms
        self.classes = y

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_names = self.file_names[idx][0]
        image = load_image_collage(img_file_names)

        class_ = self.classes[idx][0]

        if class_ == 'A':
            y_ = 0
            y_c = 0

        elif class_ == 'B':
            y_ = 0
            y_c = 1
        elif class_ == 'C':
            y_ = 1
            y_c = 2
        elif class_ == 'D':
            y_ = 1
            y_c = 3
        image = cv2.resize(image, (self.config["res_size"][0], self.config["res_size"][1])).astype(np.uint8)

        #print(np.unique(image))

        data = {"image": image}

        augmented = self.transforms(**data)
        image = augmented["image"]
        image = min_max_normalization(image)
        if len(image.shape) != 3:
            image = image[:, :, None]
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image).to(torch.float32), torch.tensor(y_).to(torch.float32), torch.tensor(y_c).to(torch.float32) #torch.from_numpy(roi).to(torch.float32), , torch.from_numpy(y_oneHot).to(torch.float32)


def load_image_collage(path):
    collage = cv2.cvtColor(cv2.imread(path.replace('/Users/ekaterinaserkova/Downloads','/home/sysadmin/autogradingML')),
                           cv2.COLOR_BGR2RGB)
    return collage



def load_image(path):
    #print(path)
    img = []
    phone_mask = []
    buttons_mask = []
    damage_mask = []
    list_names = glob(path + '/*')
    #print(path+ '/*', list_names)
    for name_ in list_names:
        #if '_fr' in name_:
            #print('name', name_)
            if os.path.exists(name_) == False:
                print("Image file doesn`t exist")
            img.append(cv2.cvtColor(cv2.imread(name_), cv2.COLOR_BGR2RGB))
            mask_filename = '/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/Results/PhonesDataset_forClassification/' + \
                            name_.split('/')[::-1][0].replace('.jpg', '.png')
            if os.path.exists(mask_filename) == False:
                print("Mask file doesn`t exist")
            phone_mask_=cv2.cvtColor(
                cv2.imread(mask_filename), cv2.COLOR_BGR2RGB)
            phone_mask.append(phone_mask_)
            buttons_filename = '/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/Results/Buttons_forClassification/' + \
                               name_.split('/')[::-1][0].replace('.jpg', '.png')
            if os.path.exists(buttons_filename) == False:
                print("Buttons file doesn`t exist")
            buttons_mask.append(cv2.cvtColor(
                cv2.imread(buttons_filename), cv2.COLOR_BGR2RGB))

            # print(buttons_filename, cv2.cvtColor(
            #     cv2.imread(buttons_filename), cv2.COLOR_BGR2RGB).shape)
            #
            # damage_filename = '/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/datasets/Damage_forClassification_CV2/' + \
            #                   name_.split('/')[::-1][0].replace('.jpg', '.png')
            # damage_filename = '/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/Results/DamageDataset_results_forClassification/' + \
            #                    name_.split('/')[::-1][0].replace('.jpg', '.png')
            # damage_filename = '/home/sysadmin/autogradingML/phone_sem_seg/stage1_experiment/datasets/' \
            #                   'Damage_CV/' + \
            #                   name_.split('/')[::-1][0].replace('.jpg', '.png')
            # #print(damage_filename)
            # if os.path.exists(damage_filename) == False:
            #     print("Damage file doesn`t exist")
            #     print(damage_filename)
            # dam_mask = cv2.cvtColor(
            #     cv2.imread(damage_filename), cv2.COLOR_BGR2RGB)
            # print(phone_mask_.shape, dam_mask.shape)
            # phone_mask_ = cv2.erode(phone_mask_, np.ones((5, 5)), iterations=1)
            # x, y, z = np.where(phone_mask_)
            # dam_ = dam_mask
            # dam_true = np.zeros(phone_mask_.shape)
            # dam_true[min(x):max(x), min(y):max(y)] = dam_
            # dam_true = cv2.resize(dam_true, (512, 512))
            # dam_mask = dam_true
            dam_mask = np.ones((phone_mask_.shape))
            damage_mask.append(dam_mask)

    img = np.array(img)
    phone_mask = np.array(phone_mask)
    buttons_mask = np.array(buttons_mask)
    damage_mask = np.array(damage_mask)
    #print( damage_mask.shape)
    #print(list_names, img.shape, phone_mask.shape, buttons_mask.shape,  np.unique(damage_mask))
    return img.astype(np.float32), phone_mask.astype(np.uint8), buttons_mask.astype(np.uint8), damage_mask.astype(np.uint8)

