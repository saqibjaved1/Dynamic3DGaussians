
import torch
import numpy as np
import open3d as o3d
import time
import copy
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult
from external import build_rotation
from colormap import colormap
from copy import deepcopy
from PIL import Image
import json
import torchvision
import os
from os import makedirs
from tqdm import tqdm
import torchvision.transforms.functional as tf
from lpipsPyTorch import lpips

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def load_scene_data(seq, exp, masking=False, masking_method="ste", prune=False, pruning_method="avg", seg_as_col=False, threshold=0.01):
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    indices = None
    if prune:
        masks = torch.sigmoid(params["mask"])

        print(f"Initial Gaussians: {len(masks[0])}")
        if pruning_method == "avg":
            mean_masks = masks.mean(dim=0)
            indices = mean_masks < threshold
        elif pruning_method == "any":
            indices = (masks < threshold).any(dim=0)
        elif pruning_method == "all":
            indices = (masks < threshold).all(dim=0)
        else:
            print("Pruning Method Not Implemented")
            exit(0)
        
        # Indices have size [N, 1]
        indices = indices.squeeze(1)
        
        print(f"Pruning Gaussians: {indices.sum().item()}")
        # Params have size [T, N, D]
        params_pruned = {}
        for k, v in params.items():
            if len(v.shape) == 3:
                params_pruned[k] = v[:, ~indices, :]
            else:
                params_pruned[k] = v
                
        params = params_pruned
        is_fg = params['seg_colors'][:, 0] > 0.5
        
        print(f"Remaining Gaussians: {len(params['means3D'][0])}")


    for t in range(len(params['means3D'])):
        if not masking:
            rendervar = {
                'means3D': params['means3D'][t],
                'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
                'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
                'opacities': torch.sigmoid(params['logit_opacities']),
                'scales': torch.exp(params['log_scales']),
                'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
            }
            scene_data.append(rendervar)
        else:

            mask = torch.sigmoid(params["mask"][t])
            
            if masking_method == "ste":
                mask = ((mask > threshold).float() - mask).detach() + mask
            elif masking_method == "sigmoid":
                pass
            else:
                print("Masking Method Not Implemented")
                exit(0)
                
            rendervar = {
                'means3D': params['means3D'][t],
                'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
                'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
                'opacities': torch.sigmoid(params['logit_opacities']) * mask,
                'scales': torch.exp(params['log_scales']) * mask,
                'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
            }
            scene_data.append(rendervar)
            
    return scene_data, is_fg

# def render(w2c, k, timestep_data):
#     with torch.no_grad():
#         cam = setup_camera(w, h, k, w2c, near, far)
#         im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
#         return im, depth
    
def render(cam, timestep_data):
    with torch.no_grad():
        im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth

def get_dataset(t, md, seq):
    dataset = []
    # print("Reading Cameras")
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"/scratch/cvlab/datasets/datasets_ahmad/panoptic/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        # seg = np.array(copy.deepcopy(Image.open(f"/scratch/cvlab/datasets/datasets_ahmad/panoptic/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        # seg = torch.tensor(seg).float().cuda()
        # seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'id': c})
    return dataset

def test(seq, exp, masking=False, masking_method="ste", prune=False, pruning_method="avg", threshold=0.01, save_prefix=""):
    print(f"Testing Experiment: {exp}, {seq}")
    
    print("Loading Metadata")
    md = json.load(open(f"/scratch/cvlab/datasets/datasets_ahmad/panoptic/{seq}/test_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])

    print("Loading Gaussian Model")
    scene_data, is_fg = load_scene_data(seq, exp, masking=masking, masking_method=masking_method, prune=prune, pruning_method=pruning_method, threshold=threshold)

    render_path = f"./output/{exp}/{seq}/renders"
    gts_path = f"./output/{exp}/{seq}/gt"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)


    ssims = []
    psnrs = []
    lpipss = []

    print("Starting Testing...")
    for t in tqdm(range(num_timesteps)):
        dataset = get_dataset(t, md, seq)
        # print("Testing")
        for camera in dataset:
            idx = camera['id']
            im, depth = render(camera["cam"], scene_data[t])
            gt = camera['im']
            
            torchvision.utils.save_image(im, os.path.join(render_path, 'exp_{save_prefix}_t_{0:05d}_{0:05d}'.format(t, idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, 'exp_{save_prefix}_t_{0:05d}_{0:05d}'.format(t, idx) + ".png"))
            ssims.append(ssim(im, gt))
            psnrs.append(psnr(im, gt).mean())
            lpipss.append(lpips(im, gt, net_type='vgg'))
            
            break
            
        break

    
    print("  # Gaussians: {}".format(len(scene_data[t]['means3D'])))
    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))





if __name__ == "__main__":
    print("Testing...")
    # print("Pretrained")
    # exp_name = "pretrained"
    # for sequence in ["basketball", "juggle", "tennis"]:
    #     test(sequence, exp_name, masking=False, prune=False)

    # print("Trained with Mask Test Without")
    # exp_name = "baseline_mask"
    # for sequence in ["basketball", "juggle", "tennis"]:
    #     test(sequence, exp_name, masking=False, prune=False)

#     print("Trained with Mask Test With STE")
#     exp_name = "baseline_mask"
#     for sequence in ["basketball", "juggle", "tennis"]:
#         test(sequence, exp_name, masking=True, masking_method="ste", prune=False , threshold=0.01, save_prefix="mask_ste_0.01")

#     print("Trained with Mask Test With Sigmoid")
#     exp_name = "baseline_mask"
#     for sequence in ["basketball", "juggle", "tennis"]:
#         test(sequence, exp_name, masking=True, masking_method="sigmoid", prune=False, threshold=0.01, save_prefix="mask_sigmoid_0.01")

    print("Trained with Mask Test without and Pruning with Avg")
    exp_name = "baseline_mask"
    for sequence in ["basketball", "juggle", "tennis"]:
        test(sequence, exp_name, masking=False, prune=True, pruning_method="avg", threshold=0.01, save_prefix="no_mask_prune_avg_0.01")

    print("Trained with Mask Test without and Pruning with Any")
    exp_name = "baseline_mask"
    for sequence in ["basketball", "juggle", "tennis"]:
        test(sequence, exp_name, masking=False, prune=True, pruning_method="any", threshold=0.01, save_prefix="no_mask_prune_any_0.01")

    print("Trained with Mask Test without and Pruning with All")
    exp_name = "baseline_mask"
    for sequence in ["basketball", "juggle", "tennis"]:
        test(sequence, exp_name, masking=False, prune=True, pruning_method="all", threshold=0.01, save_prefix="no_mask_prune_all_0.01")

    # exp_name = "baseline_mask"
    # for sequence in ["basketball", "juggle", "tennis"]:
    #     test(sequence, exp_name, masking=True, prune=True)

