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
from datasets import load_dataset
import sys, os
# sys.path.append(os.getcwd())
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

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split="train",
    image_key="image",
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms.extend([transforms.ToTensor(), 
                             transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    tform = transforms.Compose(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds

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
    

class Material(Dataset):
    """ The Dataset of text-texture pair
        Args - data_dir : path of data
             - mode : mode of the dataset, e.g. 'train'/'test'
             -image_transforms : the transforms preprocessed on the image data
    """
    def __init__(self, 
                 data_dir: str, 
                 mode: str = "train",
                 image_transforms: list = []):
        super(Material).__init__()
        self._data_dir = os.path.join(data_dir, mode)
        self._mode = mode
        self._tform = self._set_transforms(image_transforms)
        self._data = self._process()
        self._dataset_length = len(self._data)
    
    def _process(self) -> list:
        samples_names = os.listdir(self._data_dir)
        assert len(samples_names) > 0
        data = []
        for name in samples_names:
            # text
            row_text = name
            keys = row_text.split('_')
            core_key = str()
            for ch in keys[0]:
                if not isdigit(ch):
                    core_key += ch
            str_checker = core_key.lower()
            text = "A texture map of " + ' '.join([key for key in keys[1:] if str_checker.find(key) == -1]) + " {}".format(core_key)
            
            # image
            image_path = os.path.join(self._data_dir, name, "render_o.png")
            assert os.path.isfile(image_path), image_path
            image = Image.open(image_path)  # PIL.Image
            image.convert("RGB")

            data.append([text, image])
        # data = data[:-20] if self._mode == "train" else data[-20:]
        return data
    
    def _set_transforms(self, img_transforms: list = []) -> transforms:
        img_transforms = [instantiate_from_config(tt) for tt in img_transforms]
        img_transforms.extend([transforms.ToTensor(),   # row_data->(0, 1.0)
                            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])  # (0, 1.0)->(-1.0, 1.0)
        img_transforms = transforms.Compose(img_transforms)
        return img_transforms
    
    def _im_process(self, image: Image) -> Image:
        return self._tform(image)
    
    def __len__(self) -> int:
        return self._dataset_length
    
    def __getitem__(self, idx: int) -> dict:
        text, image = self._data[idx]
        image = self._im_process(image)
        return {"text": text, "image": image}
    
    
if __name__ == "__main__":
    dataset = Material("/root/hz/DataSet/material", "test")
    for i in range(len(dataset)):
        print(dataset[i]["text"], ' ', type(dataset[i]["image"]), ' ', dataset[i]["image"].shape)
        assert i < 10
        