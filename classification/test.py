import os
import json
import tqdm
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model import MInterface
from data.material import Material
from utils import load_model_path_by_args


def main(args):
    pl.seed_everything(args.seed)
    device = f'cuda:{args.device}'
    ckpt_path = load_model_path_by_args(args)
    assert ckpt_path, "ckpt_path is None !!!"
    config = OmegaConf.load(f"{args.config}")
    args = vars(args)
    for k in args.keys():
        if config.get(k, None):
            config.pop(k)
    args.update(config)
    with open('class.json', 'rb') as json_file:
        class_dict = json.load(json_file)
    
    ## model
    model = MInterface(**args)
    sd = torch.load(ckpt_path)['state_dict']
    m, u = model.load_state_dict(sd)
    if len(m) > 0:
        print("missing keys:")
        print(m)
    if len(u) > 0:
        print("unexpected keys:")
        print(u)
    model = model.model
    model.eval()
    model = model.to(device)
    normalize = torch.nn.Sigmoid().to(device)
    ## data
    dataset = Material(**args)
    dataset = DataLoader(dataset, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)

    for data in tqdm.tqdm(dataset, desc="Classifying"):
        input = data['input']
        name = data['name']
        input = input.to(device)
        with torch.no_grad():
            out = model(input)
        result = normalize(out).cpu().detach().numpy().tolist()
        patterns = []
        for idx, _ in enumerate(result):
            for i, score in enumerate(result[idx]):
                if score > 0.5:
                    patterns.append(class_dict[str(i)] + '\n')
            txt_path = os.path.join(args['data_dir'], name[idx], 'pattern.txt')
            with open(txt_path, 'w') as f:
                f.writelines(patterns)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Control
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--seed', default=23, type=int)
    parser.add_argument('--device', default=0, type=int)

    # Model Path
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)
    parser.add_argument('--config', default=None, type=str)

    # Data Info
    parser.add_argument('--dataset', default='material', type=str)
    parser.add_argument('--data_dir', default='/root/hz/DataSet/mat/train', type=str)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    main(args)
