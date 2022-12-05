# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


# class Start(data.Dataset):
#     def __init__(self, 
#                  data_dir=r'/root/hz/DataSet/dtd/images',
#                  class_num=47,
#                  train=True,
#                  no_augment=True,
#                  aug_prob=0.5,
#                  img_mean=(0.485, 0.456, 0.406),
#                  img_std=(0.229, 0.224, 0.225)):
#         # Set all input args as attributes
#         self.__dict__.update(locals())
#         self.aug = train and not no_augment

#         self.check_files()

#     def check_files(self):
#         # This part is the core code block for load your own dataset.
#         # You can choose to scan a folder, or load a file list pickle
#         # file, or any other formats. The only thing you need to gua-
#         # rantee is the `self.path_list` must be given a valid value. 
#         file_list_path = op.join(self.data_dir, 'file_list.pkl')
#         with open(file_list_path, 'rb') as f:
#             file_list = pkl.load(f)

#         fl_train, fl_val = train_test_split(
#             file_list, test_size=0.2, random_state=2333)
#         self.path_list = fl_train if self.train else fl_val

#         label_file = './data/ref/label_dict.pkl'
#         with open(label_file, 'rb') as f:
#             self.label_dict = pkl.load(f)

#     def __len__(self):
#         return len(self.path_list)

#     def to_one_hot(self, idx):
#         out = np.zeros(self.class_num, dtype=float)
#         out[idx] = 1
#         return out

#     def __getitem__(self, idx):
#         path = self.path_list[idx]
#         filename = op.splitext(op.basename(path))[0]
#         img = np.load(path).transpose(1, 2, 0)

#         labels = self.to_one_hot(self.label_dict[filename.split('_')[0]])
#         labels = torch.from_numpy(labels).float()

#         trans = torch.nn.Sequential(
#             transforms.RandomHorizontalFlip(self.aug_prob),
#             transforms.RandomVerticalFlip(self.aug_prob),
#             transforms.RandomRotation(10),
#             transforms.RandomCrop(128),
#             transforms.Normalize(self.img_mean, self.img_std)
#         ) if self.train else torch.nn.Sequential(
#             transforms.CenterCrop(128),
#             transforms.Normalize(self.img_mean, self.img_std)
#         )

#         img_tensor = trans(img)

#         return img_tensor, labels, filename
    
    
class Dtd(data.Dataset):
    def __init__(self,
                 data_dir,
                 file_list_path,
                 aug_prob,
                 img_size,
                 num_classes,
                 train,
                 **kwargs
                ):
        self.data = self.process_file(data_dir, file_list_path, num_classes)
        self.augment = self.augs_function(aug_prob, img_size, train)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ] 
        )

    def process_file(self, data_dir: str, file_path: list, num_classes: int) -> list:
        assert op.isfile(file_path), "Not a file path"
        data = []
        with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    ## image name
                    image_name = line.split(' ')[0]
                    ## label indexes
                    label_idxes = []
                    for idx in line.split(' ')[1:]:
                        label_idxes.append(int(idx))
                    input_path = op.join(data_dir, image_name)
                    input = Image.open(input_path).convert("RGB")
                    label = torch.zeros(num_classes)
                    label[label_idxes] = 1
                    data.append([input, label])
                
        return data

    def augs_function(self, aug_prob: float, img_size: int, is_train: bool):            
        t = []
        if is_train:
            t.append(transforms.RandomHorizontalFlip(aug_prob))
            t.append(transforms.RandomVerticalFlip(aug_prob))
            # t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))

        t.append(transforms.Resize((img_size, img_size)))

        return transforms.Compose(t)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = self.augment(img)
        img = self.transform(img)
        
        return [img, label]
        # finally, if we use dataloader to get the data, we will get
        # {
        #     "img_path": list, # length = batch_size
        #     "target": Tensor, # shape: batch_size * num_classes
        #     "img": Tensor, # shape: batch_size * 3 * 224 * 224
        # }