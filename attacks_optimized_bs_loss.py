#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import logging
import os
import pdb
import pickle as pkl
import sys
from pathlib import Path
from unittest.mock import patch

import einops
import matplotlib
import numpy as np
import torch
import torchvision
from einops import rearrange, reduce, repeat
from matplotlib import pyplot as plt
from timm.data import resolve_data_config
from torch import autograd, nn
from torch.utils.data import DataLoader, Dataset

matplotlib.use('Agg')

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model


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
                'swin224':'swin_small_patch4_window7_224', 'swin224base': 'swin_base_patch4_window7_224', 'swin224large': 'swin_large_patch4_window7_224',
                'coat_tiny':'coat_tiny', 'coat_mini':'coat_mini',
                'beitv2_base_224': 'beitv2_base_patch16_224', 'beitv2_large_224':'beitv2_large_patch16_224','beit_base_224': 'beit_base_patch16_224', 'beit_large_224':'beit_large_patch16_224',
                'deit3_small_224':'deit3_small_patch16_224', 'deit3_medium_224':'deit3_medium_patch16_224', 'deit3_base_224': 'deit3_base_patch16_224',  'deit3_large_224':'deit3_large_patch16_224', 'deit3_huge_224_14': 'deit3_huge_patch14_224',
                'maxvit': 'maxvit_base_224', 'maxvit_large':'maxvit_xlarge_224',
                'conv2_base':'convnextv2_base', 'conv2_huge': 'convnextv2_huge', 'conv2_large':'convnextv2_large', 'convit_base': 'convit_base', 
                'convit_tiny': 'convit_tiny', 'convit_small': 'convit_small', 'robust_ps4': 'robust_resnet_dw_small_ps4', 'robust_ps8': 'robust_resnet_dw_small_ps8', 'robust_ps16': 'robust_resnet_dw_small',
                'swin_ss0': 'swin_base_patch4_window7_224_ss0', 'swin_ss1': 'swin_base_patch4_window7_224_ss1', 
                'swin_ss2': 'swin_base_patch4_window7_224_ss2', 'swin_ss3': 'swin_base_patch4_window7_224'

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

import numpy as np


def get_top_k_indices(mytensor, k):
    v, i = torch.topk(mytensor.flatten(), k)    
    mask = mytensor.ge(v[k-1])
    return mask


def MultiPatchGDAttack(model, images, labels, loss=nn.CrossEntropyLoss(reduction='none'), iterations=40, device=torch.device('cuda:0'), max_num_patches=100, clip_flag=False, bounds=[-1, 1], patch_size=16, lr=0.033, epsilon=1.0, *args, **kwargs):
    base_images = images.clone().detach().to(device)
    images = images.to(device).requires_grad_(True)
    labels = labels.to(device)
    succ = torch.zeros(images.size(0), dtype=torch.int32, device=device)
    grad_mags = {}    

    # Calculating most salient patches
    preds = model(images)
    loss_vals = loss(preds, labels)

    print(loss_vals.shape)
    bs, ch, sx, sy = images.shape
    #pdb.set_trace()  
    grad_vals = autograd.grad(loss_vals, images, grad_outputs=torch.ones_like(labels))        
    # Using einops to calculate gradients and norms
    grad_val_patches = rearrange(grad_vals[0], 'b c (l ps1) (w ps2) -> b l w c ps1 ps2', ps1=patch_size, ps2=patch_size)
    grad_val_patches_norm = torch.norm(torch.norm(grad_val_patches, dim=(4, 5), p='fro'), dim=3, p='fro')
    # We try to find minimum number of patches required to break image by checking with a fixed number of gradient updates
    k = max_num_patches
    images = base_images.clone().detach().requires_grad_(True)
    total = []
    for i in range(bs):
        get_indices_sorted_curr = get_top_k_indices(grad_val_patches_norm[i], k)           
        total.append(get_indices_sorted_curr)
    get_indices_sorted = torch.stack(total)    
    get_indices_repeat = get_indices_sorted.unsqueeze(3).unsqueeze(4).unsqueeze(5).repeat(1, 1, 1, 3, patch_size, patch_size)    
    for i in range(iterations):        
        non_succ = torch.tensor([int(not flag) for flag in succ]).to(torch.int32).to(device)        
        preds = model(images)
        loss_vals = loss(preds, labels)
        #Change that had to be done to support batch wise gradient calculation per image
        grad_vals = autograd.grad(loss_vals, images, grad_outputs=torch.ones_like(labels))
        #print(grad_vals[0].shape)
        logging.debug(grad_vals[0][0].shape)
        #pdb.set_trace()                                                    
        new_grad_val_patches = rearrange(grad_vals[0], 'b c (l ps1) (w ps2) -> b l w c ps1 ps2', ps1=patch_size, ps2=patch_size)
        fil_grad_val_patches = new_grad_val_patches * get_indices_repeat
        fil_grad_val = rearrange(fil_grad_val_patches, 'b l w c ps1 ps2 -> b c (l ps1) (w ps2)')                       
        #print(fil_grad_val)
        images.data += (lr * (fil_grad_val.sign())) * ((non_succ).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, ch, sx, sy))
        # Infinity norm constraint. Here, we are constructing a mixed norm attack
        if clip_flag:
            images = torch.max(torch.min(images, base_images + epsilon), base_images - epsilon)
            images.clamp_(bounds[0], bounds[1])
        #pdb.set_trace()
        with torch.no_grad():
            preds2 = model(images)
            broken_indices = (torch.argmax(preds2, dim=1) != labels)
            succ[broken_indices] = 1        
    return succ, base_images, images, k


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
    parser.add_argument('-eps', '--epsilon', help='Epsilon bound for mixed norm attacks', default=1.0)
    #parser.add_argument('-ns', '--skipimages', help='No. of images to skip', default=20, type=int)
    return parser


if __name__ == '__main__':
    bs = 8
    print('entered main')
    parser = build_parser()
    args = parser.parse_args()
    print(f'config:{str(args)}')
    print('checking on outdire')
    print(args.outdir)
    y = str(args.epsilon)
    z = y.replace("/", "")
    outdir = args.outdir + "/" + str(args.mtype) +"/mt_" + str(args.mtype) + "_it_" + str(args.iteration) + "_mp_" + str(args.max_patches)  + "_ni_" + str(args.num_images) +  "_lr_" + str(args.lr) + "_ps_" + str(args.patch_size) + "_si_" + str(args.start_idx) + "_eps_" + z
    print(outdir)
    print('end checking on outdire')
    #outdir = Path(args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    mtype = args.mtype
  #  att_type = args.att_type
    clip_flag = args.clip
    epsilon = eval(str(args.epsilon))   
    print(f'epsilon value is {epsilon}')
    eps_val = epsilon

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level=logging.INFO, filename=outdir + '/run.log',
                        filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    model_name = mtype_dict[mtype]
#    logging.info(f'Running {att_type} for {model_name} for imagenet')
    if model_name is None:
        raise Exception(f'{mtype}: No such model type found')

    #model_name = 'robust_resnet_dw_small_ps4'
    if model_name=='robust_resnet_dw_small_ps4':                        
        model = create_model(model_name, pretrained=False)
        PATH = "/scratch/sca321/robustness/attacks/ameya/Transformer-attacks-master/robustcnn/RobustCNN/train_output/exp_name_ps4_fingpu4/checkpoint-289.pth.tar"
        checkpoint = torch.load(PATH,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()        
    if model_name=='robust_resnet_dw_small_ps8':                        
        model = create_model(model_name, pretrained=False)
        PATH = "/scratch/sca321/robustness/attacks/ameya/Transformer-attacks-master/robustcnn/RobustCNN/train_output/exp_name_ps8_fingpu4/checkpoint-299.pth.tar"
        checkpoint = torch.load(PATH,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()        
    if model_name=='robust_resnet_dw_small':                        
        model = create_model(model_name, pretrained=False)
        PATH = "/scratch/sca321/robustness/attacks/ameya/Transformer-attacks-master/robustcnn/RobustCNN/train_output/exp_name_ps16_fingpu4/checkpoint-299.pth.tar"
        checkpoint = torch.load(PATH,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()                        
    if model_name=='swin_base_patch4_window7_224_ss0':                        
        PATH = "/scratch/sca321/robustness/attacks/ameya/Transformer-attacks-master/swin/pytorch-image-models/train_output/exp_name_ss0_gpu4/checkpoint-281.pth.tar"
        #model = create_model(model_name, pretrained=False, checkpoint_path=PATH) 
        model = create_model(model_name, pretrained=True)    
        model.eval()
    if model_name=='swin_base_patch4_window7_224_ss1' or model_name=='swin_base_patch4_window7_224_ss2' or model_name=='swin_base_patch4_window7_224':                        
        PATH = "/scratch/sca321/robustness/attacks/ameya/Transformer-attacks-master/swin/pytorch-image-models/train_output/exp_name_ss1_gpu4/checkpoint-287.pth.tar"
        #model = create_model(model_name, pretrained=False, checkpoint_path=PATH) 
        model = create_model(model_name, pretrained=True)                                                   
        model.eval()                                
    else:
        model = create_model(model_name, pretrained=True)    
    # model.reset_classifier(num_classes=10)
    config = resolve_data_config({}, model=model)
    print(config)
    transforms = create_transform(**config)
    #cifar10_test = CIFAR10(root='./datasets/test/', download=True, train=False, transform=transforms)
    #indices = np.load('imagenet_indices.npy')
    indices = np.load('indices_10k.npy')
    imagenet_val = torch.utils.data.Subset(
        torchvision.datasets.ImageNet(root=args.dpath, split='val', transform=transforms), indices)
    test_dl = DataLoader(imagenet_val, batch_size=bs)

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
    for idx, (images, labels) in enumerate(test_dl):
        if idx*bs > args.num_images:
            break
        if idx*bs < args.start_idx:
            continue  # handling some interrupted work (temporary)
        images = images.to(device)
        labels = labels.to(device)
        bs, ch, sx, sy = images.shape
        succ_list = []
        num_patches_list = []                        
        
        succ_list, base_imges, attack_images, num_patches_list = MultiPatchGDAttack(model, images, labels, loss=nn.CrossEntropyLoss(reduction='none'
        ), iterations=args.iteration, device=device, max_num_patches=args.max_patches, clip_flag=clip_flag, bounds=bounds, patch_size=args.patch_size, lr=args.lr, epsilon=eps_val)
        logging.info(f'{idx}, {succ_list}, {num_patches_list}')
        #print(succ_list)
        attack_succ += sum(succ_list)        
        ks[idx] = (succ_list, num_patches_list)
        #ks[idx] = (succ, num_patches)

    # epsilons.append(img - clipped[])
    # sys.exit()
    import time
    time.sleep(2.4)
    print('entering the attack succ printing')
    print(attack_succ)
    
    rob_acc = 1 - attack_succ/(args.num_images+1)
    logging.info(f'Robust accuracy:{rob_acc}')
    print(f'Robust accuracy:{rob_acc}')
    file_dump = outdir + '/ks.pkl'
    with open(file_dump, 'wb') as f:
        pkl.dump(ks, f)    
    logging.shutdown()
