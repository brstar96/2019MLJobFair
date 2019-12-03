from __future__ import print_function, division
import os, cv2, torch, time
from PIL import Image
import numpy as np
import utils.custom_transforms as tr
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def splitTrainTestImgs(IMG_path, df_train, df_test):
    for i, trial in df_train.iterrows():
        df_train.loc[i, "filename"] = os.path.join(IMG_path, df_train['filename'][i])
    for i, trial in df_test.iterrows():
        df_test.loc[i, "filename"] = os.path.join(IMG_path, df_test['filename'][i])

    return df_train, df_test

def read_dataset(self):
    train_img_files = []
    train_img_paths = self.datadf['filename'].tolist()
    train_labels = self.datadf['label'].tolist()

    # 이미지 데이터만 읽기
    img_datas = []
    for image in train_img_paths:
        image = Image.open(image)
        rgb_image = image.convert("RGB")
        train_img_files.append(rgb_image)

    return train_img_files, train_labels, len(train_img_files), len(train_labels)

class JobFairDataset(Dataset):
    def __init__(self, args, mode, datadf, meta_df, input_size=128, num_workers=0, transforms=None):
        self.args = args
        self.mode = mode
        self.datadf = datadf
        self.meta_data = meta_df
        self.input_size = input_size
        self.num_workers = num_workers
        self.data, self.labels, self.length_data, self.length_labels = self.read_dataset()

        # 전처리를 위한 transforms 초기화
        self.transforms = transforms

    def __len__(self):
        return self.length_labels

    def __getitem__(self, idx):
        if self.mode == 'train':
            return {'image': self.transform_tr(self.data), 'label': self.labels}
        elif self.mode == 'val':
            return {'image' : self.transform_val(self.data), 'label' : self.labels}
        else:
            print("Invalid params input")
            raise NotImplementedError

    def read_dataset(self):
        train_img_files = []
        train_img_paths = self.datadf['filename'].tolist()
        train_labels = self.datadf['label'].tolist()

        # 이미지 데이터만 읽기
        img_datas = []
        for image in train_img_paths:
            image = Image.open(image)
            rgb_image = image.convert("RGB")
            train_img_files.append(rgb_image)

        return train_img_files, train_labels, len(train_img_files), len(train_labels)

    # custom_transforms.py의 전처리 항목을 적용한 transforms.Compose 클래스 반환
    def transform_tr(self, sample): # train data를 위한 전처리 정의
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample): # validation data를 위한 전처리 정의
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            tr.ToTensor()])

        return composed_transforms(sample)