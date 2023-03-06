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
import os.path as op
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class Material(Dataset):
    def __init__(self,
                 data_dir,
                 img_size,
                 num_classes,
                 **kwargs):
        self.data = self.process_file(data_dir, num_classes)
        self.augment = self.augs_function(img_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                transforms.Lambda(lambda x: x * 2. - 1.)
            ] 
        )

    def process_file(self, data_dir: str, num_classes: int) -> list:
        assert op.isdir(data_dir), "Not a dir"
        data = []
        sample_names = os.listdir(data_dir)
        for name in sample_names:
            sample_dir = os.path.join(data_dir, name)
            img_names = os.listdir(sample_dir)
            for img_name in img_names:
                if img_name.endswith('render_512.png'):
                    img_path = os.path.join(sample_dir, img_name)
                    input = Image.open(img_path).convert("RGB")
                    data.append([input, name])
                
        return data

    def augs_function(self, img_size: int):            
        t = []
        t.append(transforms.Resize((img_size, img_size)))

        return transforms.Compose(t)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, name = self.data[idx]
        img = self.augment(img)
        img = self.transform(img)
        
        return {'input': img, 'name': name}