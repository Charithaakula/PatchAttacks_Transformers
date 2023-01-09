#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import logging
import os
import pickle as pkl
import sys
from pathlib import Path

#import foolbox as fb
import matplotlib
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
# ,apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.models import create_model
from torch import autograd, nn
#from timm.data import Dataset, DatasetTar, create_loader, resolve_data_config, RealLabelsImagenet
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, ImageNet
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

matplotlib.use('Agg')


def get_img(x):
    tmp = x[:, ...].detach().cpu().numpy().transpose(1, 2, 0)
    return tmp


def prod(x):
    pr = 1.0
    for p in x.shape:
        pr *= p
    return pr


#coat_tiny and coat_mini are 4 patch size

mtype_dict = {'vit384': 'vit_base_patch16_384', 'vit224': 'vit_base_patch16_224',
              'wide-resnet': 'wide_resnet101_2', 'deit224': 'deit_base_patch16_224', 'bit_152_4': 'resnetv2_152x4_bitm',
              'deit224_distill':'deit_base_distilled_patch16_224', 'effnet': 'tf_efficientnet_l2_ns', 'resnet50':'resnet50', 'resnet101d':'resnet101d',
                'swin224':'swin_small_patch4_window7_224', 'swin224base': 'swin_base_patch4_window7_224', 'coat_tiny':'coat_tiny', 'coat_mini':'coat_mini',
                'beit224': 'beitv2_base_patch16_224', 'beit224_large':'beitv2_large_patch16_224','deit3_224': 'deit3_base_patch16_224', 'deit3_14_224': 'deit3_huge_patch14_224',
                'maxvit': 'maxvit_base_224', 'maxvit_large':'maxvit_xlarge_224'
                    }
#att_type_dict = {'pgdlinf': fb.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=0.033, steps=40, random_start=True),
#                 'pgdl2': fb.attacks.L2ProjectedGradientDescentAttack(steps=40, random_start=True)
#                 }


def get_patches(img, patch_size=16):
    bs, ch, sx, sy = img.size()
    patches = []
    for i in range(0, sx, patch_size):
        for j in range(0, sy, patch_size):
            patches.append(img[:, :, i:i+patch_size, j:j+patch_size])
    return patches


def reconstruct_img(patches, img):
    bs, ch, sx, sy = img.shape
    _, _, patch_size, _ = patches[0].shape
    recon = torch.zeros((bs, ch, sx, sy), device=device)
    k = 0
    for i in range(0, sx, patch_size):
        for j in range(0, sy, patch_size):
            recon[:, :, i:i+patch_size, j:j+patch_size] = patches[k]
            k += 1
    return recon


def MultiPatchGDAttack(model, img, label, loss=nn.CrossEntropyLoss(), iterations=40, device=torch.device('cuda:0'), max_num_patches=100, clip_flag=False, bounds=[-1, 1], patch_size=16, lr=0.033, epsilon=1.0, *args, **kwargs):
    base_img = copy.deepcopy(img)
    img = img.to(device)
    img.requires_grad_(True)
    bs, ch, sx, sy = img.size()
    label = label.to(device)
    print(f'epsilon:{epsilon}')
   # patch_size = 16
    #l2_norms = {}
    #max_val = 0.0
    #max_i, max_j = 0,0
    succ = 0
    grad_mags = {}
    # Calculating most salient patches
    pred = model(img)
    loss_val = loss(pred, label)
    grad_val = autograd.grad(loss_val, img)
    for i in range(0, sx, patch_size):
        for j in range(0, sy, patch_size):
            grad_mags[(i, j)] = torch.norm(
                grad_val[0][:, :, i:i+patch_size, j:j+patch_size], p="fro")
    sorted_tuples = sorted(grad_mags.items(), key=lambda x: x[1], reverse=True)
    grad_mags_sorted = [(k, v) for k, v in sorted_tuples]

    # We try to find minimum number of patches required to break image by checking with a fixed number of gradient updates
    for k in range(1, max_num_patches+1):
        img = copy.deepcopy(base_img).requires_grad_(
            True)  # Resetting image after each failure
        for i in range(iterations):
            pred = model(img)
            loss_val = loss(pred, label)
            grad_val = autograd.grad(loss_val, img)
            logging.debug(grad_val[0].shape)
            for p in range(k):
                p_x, p_y = grad_mags_sorted[p][0]
                img[:, :, p_x:p_x+patch_size, p_y:p_y+patch_size].data += (lr * \
                    (grad_val[0][:, :, p_x:p_x+patch_size, p_y:p_y+patch_size]).sign()) ## Infinity norm constraint. Here, we are constructing a mixed norm attack
            if clip_flag:
                img = torch.max(torch.min(img, img+epsilon), img-epsilon)
                img.clamp(bounds[0], bounds[1])
            with torch.no_grad():
                pred2 = model(img)
                if torch.argmax(pred2, dim=1) != label:
                    logging.info('Image broken at iteration {i}')
                    succ = 1
                    break
        if succ == 1:
            break
    return succ, base_img, img, k


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--outdir', help='Output directory', default='results/')
    #parser.add_argument('-m', '--model', help='Model path')
    parser.add_argument('-mt', '--mtype', help='Model type', choices=list(mtype_dict.keys()), default='vit224')
    parser.add_argument('-dpath', help='Data path',
                        default='/data/datasets/Imagenet/val')
    parser.add_argument('--gpu', help='gpu to use', default=0, type=int)
 #   parser.add_argument('-at', '--att_type', help='Attack type',
 #                       choices=['pgdl2', 'pgdlinf', 'gd'], default='pgdlinf')
    parser.add_argument(
        '-it', '--iteration', help='No. of iterations', type=int, default=40)
    parser.add_argument('-mp', '--max_patches',
                        help='Max number of patches allowed to be perturbed', type=int, default=20)
    parser.add_argument('-ni', '--num_images',
                        help='Number of images to be tested', default=100, type=int)
    parser.add_argument(
        '-clip', '--clip', help='Clip perturbations to original image intensities', action='store_true')
    parser.add_argument('-lr', '--lr', help='Step size',
                        type=float, default=0.033)
    parser.add_argument('-ps', '--patch_size', help='Patch size', default=16, type=int)
    parser.add_argument('-si', '--start_idx', help='Start index for imagenet', default=0, type=int)
    parser.add_argument('-eps', '--epsilon', help='Epsilon bound for mixed norm attacks', default=1.0, type=float)
    #parser.add_argument('-ns', '--skipimages', help='No. of images to skip', default=20, type=int)
    return parser


if __name__ == '__main__':
    print('entered main')
    parser = build_parser()
    args = parser.parse_args()
    print(f'config:{str(args)}')
    print('checking on outdire')
    print(args.outdir)
    outdir = args.outdir + "/mt_" + str(args.mtype) + "_it_" + str(args.iteration) + "_mp_" + str(args.max_patches)  + "_ni_" + str(args.num_images) +  "_lr_" + str(args.lr) + "_ps_" + str(args.patch_size) + "_" + str(args.start_idx) + "_eps_" + str(args.epsilon)
    print(outdir)
    print('end checking on outdire')
    #outdir = Path(args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    mtype = args.mtype
  #  att_type = args.att_type
    clip_flag = args.clip
    eps_val = args.epsilon

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level=logging.INFO, filename=outdir + '/run.log',
                        filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    model_name = mtype_dict[mtype]
#    logging.info(f'Running {att_type} for {model_name} for imagenet')
    if model_name is None:
        raise Exception(f'{mtype}: No such model type found')

    model = create_model(model_name, pretrained=True)
    # model.reset_classifier(num_classes=10)
    config = resolve_data_config({}, model=model)
    print(config)
    transforms = create_transform(**config)
    #cifar10_test = CIFAR10(root='./datasets/test/', download=True, train=False, transform=transforms)
    indices = np.load('imagenet_indices.npy')
    imagenet_val = Subset(
        ImageNet(root=args.dpath, split='val', transform=transforms), indices)
    test_dl = DataLoader(imagenet_val, batch_size=1)

    model = model.to(device)
    #sd = torch.load('cifar10_vit.pth', map_location='cuda:0')
    # model.load_state_dict(sd)
    numparams = 0
    for params in model.parameters():
        numparams += prod(params)

    model.eval()
    # TODO:Figure out a smarter way of calculating image bounds
    bounds = [(0-config['mean'][0])/config['std'][0],
              (1-config['mean'][0])/config['std'][0]]
        
    eps_val = (eps_val)/config['std'][0]
    args.lr = args.lr/config['std'][0]
    print(config, bounds, eps_val)
    clean_acc = 0.0
    for idx, (img, label) in enumerate(test_dl):
        if idx < args.start_idx:
            continue
        if idx > args.num_images:
            break
        img = img.to(device)
        #print(img.min(), img.max())
        label = label.to(device)
        pred = torch.argmax(model(img), dim=1)
        clean_acc += torch.eq(pred, label).sum()
    logging.info(f'Clean accuracy for imagenet subset:{clean_acc/(args.num_images+1)}')
    import time
    time.sleep(2.4)
    #corrects = np.zeros(8)
    #epsilons = []
    attack_succ = 0.0
    ks = {}
    for idx, (img, label) in enumerate(test_dl):
        if idx > args.num_images:
            break
        if idx < args.start_idx:
            continue  # handling some interrupted work (temporary)
        img = img.to(device)
        bs, ch, sx, sy = img.shape
        label = label.to(device)
        succ, img, attack_img, num_patches = MultiPatchGDAttack(model, img, label, loss=nn.CrossEntropyLoss(
        ), iterations=args.iteration, device=device, max_num_patches=args.max_patches, clip_flag=clip_flag, bounds=bounds, patch_size=args.patch_size, lr=args.lr, epsilon=eps_val)
        logging.info(f'{idx}, {succ}, {num_patches}')
        attack_succ += succ
        ks[idx] = (succ, num_patches)
        if True:
            eps = torch.norm(img - attack_img, p="fro")
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(131)
            img_np = get_img(img[0, ...])
            clipped_np = get_img(attack_img[0, ...])
            ax1.imshow((img_np + 1)/2.0)
            ax2 = fig.add_subplot(132)
            ax2.imshow((clipped_np+1)/2.0)
            ax3 = fig.add_subplot(133)
            ax3.imshow(np.abs(img_np - clipped_np)*100)
            ax1.tick_params(axis='both', bottom=False, left=False,
                            labelleft=False, labelbottom=False)
            ax2.tick_params(axis='both', bottom=False, left=False,
                            labelleft=False, labelbottom=False)
            ax3.tick_params(axis='both', bottom=False, left=False,
                            labelleft=False, labelbottom=False)
            # ax1.set_title(succ)
            # ax3.set_title(f'{idx}, {succ}, {num_patches}, {eps:.04f}')
            plt.savefig(f'{outdir}/{idx}_{succ}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        # epsilons.append(img - clipped[])
    # sys.exit()
    import time
    time.sleep(2.4)
    print('entering the attack succ printing')
    print(attack_succ)
    
    rob_acc = 1 - attack_succ/(args.num_images+1)
    logging.info(f'Robust accuracy:{rob_acc}')
    with open(outdir / 'ks.pkl', 'wb') as f:
        pkl.dump(ks, f)    
    logging.shutdown()
