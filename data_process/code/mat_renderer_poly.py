import os.path
import time
import sys
sys.path.append("/home/d5/hz/Code/diffmat/diffmat/core")
import cv2
import numpy as np
import torch
from PIL import Image
from render import Renderer


def processing(dir_name, root_dir, device='cuda'):
    path_pre = os.path.join(root_dir, dir_name)
    # get base data
    img_type = 'jpg'
    base_path = os.path.join(path_pre, f"_diff_4k.{img_type}")
    if not os.path.exists(base_path):
        img_type = 'png'
        base_path = os.path.join(path_pre, f"_diff_4k.{img_type}") 
        if not os.path.exists(base_path):
            return []
        
    basecolor = cv2.imread(base_path)
    ## resize
    # basecolor = cv2.resize(basecolor, [2048, 2048])
    # basecolor = cv2.resize(basecolor, [1024, 1024])
    # basecolor = cv2.resize(basecolor, [512, 512])
    # cv2.imwrite(path_pre + '/' + '_diff_512.jpg', basecolor)
    
    H, W = basecolor.shape[:-1]
    # basecolor = cv2.resize(basecolor, (512, 512))
    basecolor = np.array(basecolor[:, :, ::-1], dtype=np.float32) / 255.
    basecolor = torch.from_numpy(basecolor.transpose(2, 0, 1)).float().to(device)
    
    normal_path = os.path.join(path_pre, f"_nor_dx_4k.{img_type}") 
    if not os.path.exists(normal_path):
        return []

    normal = cv2.imread(normal_path)
    ## resize
    # normal = cv2.resize(normal, [2048, 2048])
    # normal = cv2.resize(normal, [1024, 1024])
    # normal = cv2.resize(normal, [512, 512])
    # cv2.imwrite(path_pre + '/' + '_nor_dx_512.jpg', normal)
    
    if normal.shape[0] != H or normal.shape[1] != W:
        return []
    # normal = cv2.resize(normal, (512, 512))
    normal = np.array(normal[:, :, ::-1], dtype=np.float32) / 255.
    normal = torch.from_numpy(normal.transpose(2, 0, 1)).float().to(device)

    roughness_path = os.path.join(path_pre, f"_rough_4k.{img_type}") 
    if not os.path.exists(roughness_path):
        # return []
        roughness = np.zeros((H, W, 3))
    else:
        roughness = cv2.imread(roughness_path)
        ## resize
        # roughness = cv2.resize(roughness, [2048, 2048])
        # roughness = cv2.resize(roughness, [1024, 1024])
        # roughness = cv2.resize(roughness, [512, 512])
        # cv2.imwrite(path_pre + '/' + '_rough_512.jpg', roughness)
        
    if roughness.shape[0] != H or roughness.shape[1] != W:
        return []
    # roughness = cv2.resize(roughness, (512, 512))
    roughness = np.array(roughness[:, :, ::-1], dtype=np.float32) / 255.
    roughness = torch.from_numpy(roughness.transpose(2, 0, 1)).float().to(device)
    roughness[1, :, :] = 0.0
    roughness[2, :, :] = 1.0

    metalness_path = os.path.join(path_pre, f"_metal_4k.{img_type}") 
    if os.path.exists(metalness_path):
        metalness = cv2.imread(metalness_path)
        ## resize
        # metalness = cv2.resize(metalness, [2048, 2048])
        # metalness = cv2.resize(metalness, [1024, 1024])
        # metalness = cv2.resize(metalness, [512, 512])
        # cv2.imwrite(path_pre + '/' + '_rough_512.jpg', metalness)
        
        if metalness.shape[0] != H or metalness.shape[1] != W:
            return []
        # metalness = cv2.resize(metalness, (512, 512))
        metalness = np.array(metalness[:, :, ::-1], dtype=np.float32) / 255.
        metalness = torch.from_numpy(metalness.transpose(2, 0, 1)).float().to(device)
        roughness[1, :, :] = metalness[0, :, :]

    # opacity_path = os.path.join(path_pre, id_name + f"_4K_Opacity.{}")
    # if os.path.exists(opacity_path):
    #     opacity = cv2.imread(opacity_path)
    #     if opacity.shape[0] != H or opacity.shape[1] != W:
    #         return []
    #     # roughness = cv2.resize(metalness, (512, 512))
    #     opacity = np.array(opacity[:, :, ::-1], dtype=np.float32) / 255.
    #     opacity = torch.from_numpy(opacity.transpose(2, 0, 1)).float().to(device)
    #     roughness[2, :, :] = opacity[0, :, :]

    roughness = roughness.reshape(1, 3, H, W)
    normal = normal.reshape(1, 3, H, W)
    basecolor = basecolor.reshape(1, 3, H, W)

    # a = torch.stack([basecolor, normal, roughness[:, :1, :, :], roughness[:, 1:2, :, :], roughness[:, 2:3, :, :]])
    l = (basecolor, normal, roughness[:, :1, :, :], roughness[:, 1:2, :, :], roughness[:, 2:3, :, :])
    return l

device = 'cuda:1'
r = Renderer(device=device, normal_format='dx')
data_dir = "/media/d5/7D1922F98D178B12/hz/DataSet/mat/data/polyhaven"
names = os.listdir(data_dir)
names.sort()
print(len(names))
count = 0
for i in range(0, len(names)):
    # if i < 890:
    #     continue
    dir_name = names[i]
    
    ## continue if already rendered
    # if os.path.exists(os.path.join(data_dir, dir_name, "render_512.png")):
    #     count += 1
    # continue

    ## add label text
    pattern_path = os.path.join(data_dir, dir_name, 'pattern.txt')
    if not os.path.isfile(pattern_path):
        continue
    with open(pattern_path, 'r') as pattern_f:
        patterns = pattern_f.readlines()

    pattern_list = []
    for pattern in patterns:
        pattern = pattern.strip()
        if pattern != '':
            pattern_list.append(pattern)
    
    text_wo_pattern_path = os.path.join(data_dir, dir_name, "text_wo_label.txt")
    if not os.path.isfile(text_wo_pattern_path):
        continue
    with open(text_wo_pattern_path, 'r') as f:
        text = f.read().strip()

    text = text + (" with {} pattern".format(' and '.join(pattern_list)) if len(pattern_list) > 0 else '')
    text_path = os.path.join(data_dir, dir_name, 'new_text.txt')
    with open(text_path, 'w') as f:
        f.write(text)
    
    # if not os.path.exists(text_path):
    #     continue
    # with open(text_path, 'r') as f:
    #     text = f.read().strip()
    
    # if len(text) == 0:
    #     print(i, ' ', dir_name)#, end='')
    # continue
    
    print(i, ' ', dir_name)#, end='')
    continue
    l = processing( dir_name, data_dir, device)
    if l:
        start = time.time()
        out = r.evaluate(*l)

        gt_image = out.permute((0, 2, 3, 1)).cpu().clamp_(0, 1)[0]
        np_gt_image = gt_image.numpy()

        gt_img = (np_gt_image * 255).astype(np.uint8)
        gt_img = Image.fromarray(gt_img)
        gt_img.save(os.path.join(data_dir, dir_name, "render_4K.png"))
        gt_img = gt_img.resize((512, 512))
        gt_img.save(os.path.join(data_dir, dir_name, "render_512.png"))
        end = time.time()
        print("\t{:.1f}".format(end - start))
    assert 0
print(count)