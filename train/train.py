import argparse
import os
import os.path as osp
import random
import numpy as np
import torch
from trainer import Trainer
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser()

# Active Learning Parameters
# parser.add_argument('-l', 'annotation_budget', type=int, default=1e9, help='number of annotation budget')
parser.add_argument('--plan_path', type=str, default='/media/ubuntu/FM-AL/data/AL_Plan/Heart/clip_RN50x64/FPS_109.csv', help='path of the plan file for customized selection')

# Experiment Parameters
parser.add_argument('-c', '--num_classes', type=int, required=True, help='number of classes')
# parser.add_argument('-m', '--modality', type=str, required=True, help='The input imaging modality ["ct" | "mr"]')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='networks are saved here')
parser.add_argument('--load_ckpt', help='whether to use the pre-trained model weights/checkpoint. Defaults to False')
parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('-bs', '--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--base_lr', type=float, default=1e-2, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--max_epochs', type=int, default=400, help='maximum number of epochs')
parser.add_argument('--max_iter', type=int, default=6000, help='maximum number of epochs')
parser.add_argument('--val_epochs', type=int, default=10, help='number of iterations per validation phase')
parser.add_argument('--dim', type=int, default=2, help='image dimension')
parser.add_argument('--input_nc', type=int, default=1, help='number of input image channels')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--dataset', type=str, default='CVC-ClinicDB', help='dataset name')
parser.add_argument('--base_dir', type=str, default='/media/ubuntu/FM-AL', help='base directory')
parser.add_argument('--deterministic', type=int, default=1, help="whether use deterministic training")
parser.add_argument('--img_size', type=int, default=288, help='image size for training')
                
        
args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    trainer = Trainer(args=args)
    trainer.train()
            
        