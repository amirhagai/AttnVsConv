


####################################################################
# code mostly taken from https://github.com/tanimutomo/cifar10-c-eval.git
####################################################################


import argparse
import glob
import numpy as np
import os
import pprint
import torch
import torchvision
import tqdm

from glob import glob
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.backends.cudnn as cudnn


################################################
# TODO - remove after debug

import sys
sys.path.append('/code')
################################################


from Cifar10_C.Cifar10_utils import load_txt, accuracy, create_barplot, get_fname, AverageMeter
from Cifar10_C.Cifar10 import CIFAR10C

from resnet import ResNet18, ResNet152
from choose_and_replace import replace_layer


CORRUPTIONS = load_txt('/code/Cifar10_C/corruptions.txt')
MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]




def main(opt, weight_path :str):

    device = torch.device(opt.device)

    # model
    if opt.arch == 'resnet152':
        model = ResNet152()
        ckpt_name = 'ckpt_152.pth'
    elif opt.arch == 'resnet18':
        model = ResNet18()
        ckpt_name = 'ckpt.pth'
    elif opt.arch == 'resnet18_replace':
        model = ResNet18()
        input_shape = (1, 3, 32, 32)
        replace_layer(model, 'layer4.1.conv2', input_shape)
        ckpt_name = 'ckpt_with_replace.pth'
    else:
        model = ResNet18()
        input_shape = (1, 3, 32, 32)
        replace_layer(model, opt.arch[len('ckpt_'):], input_shape)
        ckpt_name = f'{opt.arch}.pth'        
    
    weight_path = os.path.join(weight_path, ckpt_name)
    
    model = model.to(opt.device)
    if opt.device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    try:
        checkpoint = torch.load(weight_path, map_location=opt.device)['net']
        if opt.device == 'cpu':
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    except:
        checkpoint = torch.load(weight_path, map_location=opt.device)['net']
        if opt.device == 'cpu':
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    # model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    accs = dict()
    with tqdm(total=len(opt.corruptions), ncols=80) as pbar:
        for ci, cname in enumerate(opt.corruptions):
            print(cname)
            # load dataset
            if cname == 'natural':
                dataset = datasets.CIFAR10(
                    os.path.join(opt.data_root, 'cifar10'),
                    train=False, transform=transform, download=True,
                )
            else:
                dataset = CIFAR10C(
                    os.path.join(opt.data_root, 'CIFAR-10-C'),
                    cname, transform=transform
                )
            loader = DataLoader(dataset, batch_size=opt.batch_size,
                                shuffle=False, num_workers=4)
            acc_meter = AverageMeter()
            with torch.no_grad():
                for itr, (x, y) in enumerate(loader):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, dtype=torch.int64, non_blocking=True)

                    z = model(x)
                    loss = F.cross_entropy(z, y)
                    acc, _ = accuracy(z, y, topk=(1, 5))
                    acc_meter.update(acc.item())

            accs[f'{cname}'] = acc_meter.avg

            pbar.set_postfix_str(f'{cname}: {acc_meter.avg:.2f}')
            pbar.update()
    
    avg = np.mean(list(accs.values()))
    accs['avg'] = avg

    pprint.pprint(accs)
    save_name = get_fname(weight_path)
    create_barplot(
        accs, save_name + f' / avg={avg:.2f}',
        os.path.join(opt.fig_dir, save_name+'.png')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--arch',
        type=str, default='ckpt_layer1.0.conv1',
        help='model name'
    )
    parser.add_argument(
        '--weight_dir',
        type=str,
        help='path to the dicrectory containing model weights',
        default='/code/checkpoint'
    )
    parser.add_argument(
        '--weight_path',
        type=str,
        help='path to the dicrectory containing model weights',
        default='/code/checkpoint'
    )
    parser.add_argument(
        '--fig_dir',
        type=str, default='/code/Figs',
        help='path to the dicrectory saving output figure',
    )
    parser.add_argument(
        '--data_root',
        type=str, default='/code/data',
        help='root path to cifar10-c directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int, default=1024,
        help='batch size',
    )
    parser.add_argument(
        '--corruptions',
        type=str, nargs='*',
        default=CORRUPTIONS,
        help='testing corruption types',
    )
    parser.add_argument(
        '--gpu_id',
        type=str, default=0,
        help='gpu id to use'
    )
    
    parser.add_argument(
        '--device',
        type=str, default='cpu',
        help='device to use'
    )

    opt = parser.parse_args()

    if opt.weight_path is not None:
        main(opt, opt.weight_path)
    elif opt.weight_dir is not None:
        for path in glob(f'./{opt.weight_dir}/*.pth'):
            print('\n', path)
            main(opt, path)
    else:
        raise ValueError("Please specify weight_path or weight_dir option.")