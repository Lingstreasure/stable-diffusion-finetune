from cgi import test
from curses import endwin
from curses.ascii import isdigit
from genericpath import isdir
from tokenize import ContStr
from typing import List
from xmlrpc.client import DateTime
import PIL
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from transformers import Data2VecTextConfig
import sys, os
sys.path.append(os.getcwd())
from ldm import data
from ldm.util import instantiate_from_config


class FolderData(Dataset):
    def __init__(self, root_dir, caption_file=None, image_transforms=[], ext="jpg") -> None:
        self.root_dir = Path(root_dir)
        self.default_caption = ""
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                captions = json.load(f)
            self.captions = captions
        else:
            self.captions = None

        self.paths = list(self.root_dir.rglob(f"*.{ext}"))
        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms

        # assert all(['full/' + str(x.name) in self.captions for x in self.paths])

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions[chosen]
            if caption is None:
                caption = self.default_caption
            im = Image.open(self.root_dir/chosen)
        else:
            im = Image.open(self.paths[index])

        im = self.process_im(im)
        data = {"image": im}
        if self.captions is not None:
            data["txt"] = caption
        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]
    

class Text2Material(Dataset):
    """ The Dataset of text-texture pair
        Args - data_dir : path of data
             - mode : mode of the dataset, e.g. 'train'/'test'
             -image_transforms : the transforms preprocessed on the image data
    """
    def __init__(self, 
                 data_dir: str, 
                 mode: str = "train",
                 image_transforms: list = []):
        super(Text2Material).__init__()
        self._data_dir = os.path.join(data_dir, mode)
        self._mode = mode
        self._tform = self._set_transforms(image_transforms)
        self._data = self._preprocess()
        self._dataset_length:int = len(self._data)
        assert self._dataset_length > 0
        print("Data Num : {} for {}".format(self._dataset_length, mode))
    
    def _preprocess(self) -> list:
        samples_names = os.listdir(self._data_dir)
        assert len(samples_names) > 0
        data = []
        length = len(samples_names)
        for name in samples_names:
            sample_dir = os.path.join(self._data_dir, name)
            ## text
            # row_text = name
            # keys = row_text.split('_')
            # core_key = str()
            # for ch in keys[0]:
            #     if not isdigit(ch):
            #         core_key += ch
            # str_checker = core_key.lower()
            # text = "A texture map of " + ' '.join([key for key in keys[1:] if str_checker.find(key) == -1]) + " {}".format(core_key)
            text_path = os.path.join(sample_dir, 'text_wo_label.txt')
            if not os.path.isfile(text_path):
                continue
            with open(text_path, 'r') as f:
                text = f.read().strip()
            ## image
            image_path = os.path.join(self._data_dir, name, 'render_small.png')
            if not os.path.isfile(image_path):
                continue
            image = Image.open(image_path)  # PIL.Image
            image.convert("RGB")  # h w c
            # if self._mode == "test":
            #     image = np.zeros(np.array(image).shape, dtype=np.array(image).dtype)

            data.append([text, image])
        return data
    
    def _set_transforms(self, img_transforms: list = []) -> transforms:
        img_transforms = [instantiate_from_config(tt) for tt in img_transforms]
        img_transforms.extend([transforms.ToTensor(),   # row_data->(0, 1.0), h w c -> c h w
                            #    transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c')), used for stable-diffusion
                               transforms.Lambda(lambda x: x * 2. - 1.)])  # (0, 1.0)->(-1.0, 1.0)
        img_transforms = transforms.Compose(img_transforms)
        return img_transforms
    
    def _img_process(self, image: Image) -> Image:
        return self._tform(image)
    
    def __len__(self) -> int:
        return self._dataset_length
    
    def __getitem__(self, idx: int) -> dict:
        text, image = self._data[idx]
        image = self._img_process(image)
        return {"txt": text, "image": image}
    
    
class PBRMap(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 mode: str = "train",
                 pbr_type: str = "albedo",
                 image_transforms: list = []):
        """The Dataset of pbr map for training decoder of VAE

        Args:
            data_dir (str): path of data
            mode (str, optional): mode of the dataset, e.g. "train"/"test". Defaults to "train".
            pbr_type (str, optional): the tyep of pbr map, e.g. "albedo"/"normal". Defaults to "albedo".
            image_transforms (list, optional): the transforms preprocessed on the image. Defaults to [].
        """             
        super().__init__()
        self._data_dir = os.path.join(data_dir, mode)
        self._mode = mode
        self.pbr_type = pbr_type
        self._tform = self._set_transforms(image_transforms)
        self._data = self._preprocess()
        self._dataset_length:int = len(self._data)
        assert self._dataset_length > 0
        # print("Data Num : {} for {}".format(self._dataset_length, mode))
    
    def _preprocess(self) -> list:
        samples_names = os.listdir(self._data_dir)
        assert len(samples_names) > 0
        data = []
        gt_postfix = "normal.png" if self.pbr_type == "normal" else "albedo.png"
        for i, name in enumerate(samples_names):
            ## text
            row_text = name
            keys = row_text.split('_')
            ## images 
            gt_path = os.path.join(self._data_dir, name, "samll_" + gt_postfix)
            input_path = os.path.join(self._data_dir, name, 'render_o.png')
            if not os.path.isfile(input_path) or not os.path.isfile(gt_path):
                continue
            input = Image.open(input_path).convert("RGB")  # PIL.Image，h w c
            input = np.array(input)
            
            gt = Image.open(gt_path).convert("RGB")
            # gt = gt.resize(input.shape[:-1])  # 1024 -> 512 already processed
            gt = np.array(gt)
            
            data.append([input, gt])
        return data
    
    def _set_transforms(self, img_transforms: list = []) -> transforms:
        img_transforms = [instantiate_from_config(tt) for tt in img_transforms]
        img_transforms.extend([transforms.ToTensor(),   # row_data->(0, 1.0), h w c -> c h w
                            #    transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c')), used for stable-diffusion
                               transforms.Lambda(lambda x: x * 2. - 1.)])  # (0, 1.0)->(-1.0, 1.0)
        img_transforms = transforms.Compose(img_transforms)
        return img_transforms
    
    def _img_process(self, image: Image) -> Image:
        return self._tform(image)
    
    def __len__(self) -> int:
        return self._dataset_length
    
    def __getitem__(self, idx: int) -> dict:
        input, gt = self._data[idx]
        # images = np.concatenate([input, gt], axis=-1)  # concat at channel dim
        # images = self._img_process(images)
        input = self._img_process(input)
        gt = self._img_process(gt)
        return {"input": input, "gt": gt}
    
if __name__ == "__main__":
    dataset = PBRMap("/root/hz/DataSet/mat", "train")
    for i in range(len(dataset)):
        print(dataset[i]["input"].shape, ' ', dataset[i]["gt"].shape)
        print(torch.min(dataset[i]["input"]), ' ', torch.max(dataset[i]["input"]))
        print(torch.min(dataset[i]["gt"]), ' ', torch.max(dataset[i]["gt"]))
        
        