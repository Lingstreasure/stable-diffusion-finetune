# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
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

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model import MInterface
from data import DInterface
from utils import load_model_path_by_args


def load_callbacks(ckptdir: str):
    callbacks = []
    # callbacks.append(plc.EarlyStopping(
    #     monitor='val_mAP',
    #     mode='max',
    #     patience=10,
    #     min_delta=0.001
    # ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_mAP',
        dirpath=ckptdir,
        filename='best-{epoch:02d}-{val_mAP:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    log_root_dir = "/root/hz/Code/stable-diffusion-finetune/classification/logs"
    logdir = os.path.join(log_root_dir, args.log_name)
    os.makedirs(logdir, exist_ok=True)
    
    
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))
    args.load_path = load_path

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        # import torch        
        # state_dict = torch.load(load_path)
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith('classifier'):
        #         continue
        #     new_state_dict['model.' + k] = state_dict[k]
        # del state_dict
        # save_state_dict = {'state_dict': new_state_dict}
        # print(save_state_dict.keys())
        # print(new_state_dict.keys())
        # torch.save(save_state_dict, os.path.join(os.path.dirname(load_path), 'resnet101cut.pth'))
        # args.resume_from_checkpoint = load_path

    # # If you want to change the logger's saving folder
    logger = TensorBoardLogger(save_dir=log_root_dir, name=args.log_name)
    ckptdir = os.path.join(logger.log_dir, "checkpoints")
    print('######################')
    print(logger.log_dir.split('/')[-1])
    os.makedirs(ckptdir, exist_ok=True)
    args.callbacks = load_callbacks(ckptdir)
    args.logger = logger

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    # parser.add_argument('--strategy', default='dp', choices=['dp', 'ddp', 'ddp2'], type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--seed', default=23, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', default=None, choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=100, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='dtd', type=str)
    parser.add_argument('--data_dir', default='/root/hz/DataSet/dtd/images', type=str)
    parser.add_argument('--train_list_path', type=str)
    parser.add_argument('--test_list_path', type=str)
    
    parser.add_argument('--model_name', default='resnet_backbone', 
                        choices=['resnet_backbone', 'resnet_csra', 'vit', 'vit_b16_224_csra', 'vit_l16_224_csra'], type=str)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam",default=0.1, type=float)
    parser.add_argument('--depth', default=50, choices=[18, 34, 50, 101, 152],type=int)
    parser.add_argument('--num_classes', default=47, type=int)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--cutmix', default=None, type=str)
    
    parser.add_argument('--loss', default='multi', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_name', default='', type=str)
    
    # # Model Hyperparameters
    # parser.add_argument('--hid', default=64, type=int)
    # parser.add_argument('--block_num', default=8, type=int)
    # parser.add_argument('--in_channel', default=3, type=int)
    # parser.add_argument('--layer_num', default=5, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    # parser.set_defaults(max_epochs=100)

    args = parser.parse_args()
    args.max_epochs = args.epochs
    # List Arguments
    # args.mean_sen = [0.485, 0.456, 0.406]
    # args.std_sen = [0.229, 0.224, 0.225]

    main(args)
