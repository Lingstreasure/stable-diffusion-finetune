import os.path
import time
import sys
sys.path.append("/root/hz/Code/diffmat/diffmat/core")
import cv2
import numpy as np
import torch
from PIL import Image
from render import Renderer


def processing(id_name, dir_name, root_dir, device='cuda'):
    path_pre = os.path.join(root_dir, dir_name)
    # get base data
    img_type = 'jpg'
    base_path = os.path.join(path_pre, id_name + f"_Color.{img_type}")
    if not os.path.exists(base_path):
        img_type = 'png'
        base_path = os.path.join(path_pre, id_name + f"_Color.{img_type}") 
        if not os.path.exists(base_path):
            return []
        
    basecolor = cv2.imread(base_path)
    H, W = basecolor.shape[:-1]
    H_2, W_2 = H // 2, W // 2
    H_4, W_4 = H // 4, W // 4
    
    ## resize
    # if H_2 >= 512 or W_2 >= 512:
    #     basecolor = cv2.resize(basecolor, [H_2, W_2])
    # if H_4 >= 512 or W_4 >= 512:        
    #     basecolor = cv2.resize(basecolor, [H_4, W_4])
    # if basecolor.shape[0] != 512:
    #     basecolor = cv2.resize(basecolor, [512, 512])
    # cv2.imwrite(os.path.join(path_pre, "basecolor_512.jpg"), basecolor)
    
    # basecolor = cv2.resize(basecolor, (512, 512))
    basecolor = np.array(basecolor[:, :, ::-1], dtype=np.float32) / 255.
    basecolor = torch.from_numpy(basecolor.transpose(2, 0, 1)).float().to(device)

    normal_path = os.path.join(path_pre, id_name + f"_NormalDX.{img_type}") 
    if not os.path.exists(normal_path):
        return []

    normal = cv2.imread(normal_path)
    if normal.shape[0] != H or normal.shape[1] != W:
        return []
    
    ## resize
    # if H_2 >= 512 or W_2 >= 512:
    #     normal = cv2.resize(normal, [H_2, W_2])
    # if H_4 >= 512 or W_4 >= 512:        
    #     normal = cv2.resize(normal, [H_4, W_4])
    # if normal.shape[0] != 512:
    #     normal = cv2.resize(normal, [512, 512])
    # cv2.imwrite(os.path.join(path_pre, "normal_512.jpg"), normal)
    
    # normal = cv2.resize(normal, (512, 512))
    normal = np.array(normal[:, :, ::-1], dtype=np.float32) / 255.
    normal = torch.from_numpy(normal.transpose(2, 0, 1)).float().to(device)

    roughness_path = os.path.join(path_pre, id_name + f"_Roughness.{img_type}") 
    if not os.path.exists(roughness_path):
        # return []
        roughness = np.zeros((H, W, 3))
        # roughness = np.zeros((512, 512, 3))
    else:
        roughness = cv2.imread(roughness_path)
        ## resize
        # if H_2 >= 512 or W_2 >= 512:
        #     roughness = cv2.resize(roughness, [H_2, W_2])
        # if H_4 >= 512 or W_4 >= 512:        
        #     roughness = cv2.resize(roughness, [H_4, W_4])
        # if roughness.shape[0] != 512:
        #     roughness = cv2.resize(roughness, [512, 512])
        # cv2.imwrite(os.path.join(path_pre, "roughness_512.jpg"), roughness)
        
    if roughness.shape[0] != H or roughness.shape[1] != W:
        return []
    # roughness = cv2.resize(roughness, (512, 512))
    roughness = np.array(roughness[:, :, ::-1], dtype=np.float32) / 255.
    roughness = torch.from_numpy(roughness.transpose(2, 0, 1)).float().to(device)
    roughness[1, :, :] = 0.0
    roughness[2, :, :] = 1.0

    metalness_path = os.path.join(path_pre, id_name + f"_Metallic.{img_type}") 
    if os.path.exists(metalness_path):
        metalness = cv2.imread(metalness_path)
        if metalness.shape[0] != H or metalness.shape[1] != W:
            return []
        
        ## resize
        # if H_2 >= 512 or W_2 >= 512:
        #     metalness = cv2.resize(metalness, [H_2, W_2])
        # if H_4 >= 512 or W_4 >= 512:        
        #     metalness = cv2.resize(metalness, [H_4, W_4])
        # if metalness.shape[0] != 512:
        #     metalness = cv2.resize(metalness, [512, 512])
        # cv2.imwrite(os.path.join(path_pre, "metalness_512.jpg"), metalness)
        
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

device = 'cuda'
light_color = [3300.0, 3300.0, 3300.0]
r = Renderer(device=device, light_color=light_color, normal_format='dx')
data_dir = "/root/hz/DataSet/mat/data/3dtextures"
names = os.listdir(data_dir)
names.sort()
## unify the file name
# change_dict = {"ambientOcclusion": "AmbientOcclusion", 
#                "basecolor": "Color", 
#                "BaseColor": "Color",
#                "height": "Height", 
#                "normal": "Normal", 
#                "roughness": "Roughness",
#                "Roughtness": "Roughness", 
#                "COLOR": "Color", 
#                "DISP": "Displacement", 
#                "NRM": "Normal", 
#                "OCC": "AmbientOcclusion", 
#                "SPEC": "Specular", 
#                "ROUGH": "Roughtness", 
#                "NORM": "Normal", 
#                "AO": "AmbientOcclusion", 
#                "opacity": "Opacity", 
#                "metallic": "Metallic"}
# for i, name in enumerate(names):
#     path = os.path.join(data_dir, name)
#     elements = os.listdir(path)
#     elem_names = name.split('-')
#     for j, key in enumerate(elem_names):
#         elem_names[j] = key[0].upper() + key[1:]
#     name_pre = '_'.join(elem_names)
#     # print(i, '\t', name)
#     for elem in elements:
#         if elem.startswith("Material") or elem.startswith("render"):
#             continue
#         elem_type = elem.split('.')[-1]
#         last_key = elem.split('.')[0].split('_')[-1]
#         if len(last_key) == 1:
#             last_key = '_'.join(elem.split('.')[0].split('_')[-2:])
#         new_last_key = str()
#         if last_key in change_dict.keys():
#             new_last_key = change_dict[last_key]
#         else:
#             new_last_key = last_key[0].upper() + last_key[1:]
#         # new_elem = elem.replace(last_key, new_last_key)
#         new_elem = name_pre + '_' + new_last_key.replace(' ', '') + ".{}".format(elem_type)
#         if new_elem == elem:
#             continue
#         elem_path = os.path.join(path, elem)
#         new_elem_path = os.path.join(path, new_elem)
#         print(elem_path, '\t', new_elem_path)
#         os.system(f"mv {elem_path} {new_elem_path}")
# assert 0 
print(len(names))
count = 0
# tar_no_rough_dir = "/media/d5/7D1922F98D178B12/hz/DataSet/mat/not_used/no_roughness/3dtextures"
for i in range(0, len(names)):
    # if i < 890:
    #     continue
    dir_name = names[i]
    # if dir_name != "mud-001":
    #     continue
    elem_names = dir_name.split('-')
    for j, key in enumerate(elem_names):
        elem_names[j] = key[0].upper() + key[1:]
    id_name = '_'.join(elem_names)
    
    ## continue if already rendered
    # if not os.path.exists(os.path.join(data_dir, dir_name, "roughness_512.jpg")):
    #     # count += 1
    #     print(i, ' ', dir_name)
    #     os.system("mv {} {}".format(os.path.join(data_dir, dir_name), 
    #                                 tar_no_rough_dir))
    #     # assert 0
    # continue
    
    ## re render for wrong normal
    # img_types = ['jpg', 'png', 'jpeg']
    # need_render = True
    # for type in img_types:
    #     if os.path.exists(os.path.join(data_dir, dir_name, id_name + f"_Normal.{type}")):
    #         need_render = False
    #         break
    # if not need_render:
    #     continue

    ## unify the normal
    # img_types = ['jpg', 'png', 'jpeg']
    # normal_path = os.path.join(data_dir, dir_name, id_name + "_Normal.{}")
    # for type in img_types:
    #     path = normal_path.format(type)
    #     if os.path.exists(path.replace('Normal', 'NormalDX')):
    #         continue
    #     if os.path.exists(path):
    #         print(i, ' ', dir_name)
    #         normal = cv2.imread(path)
    #         normal = np.asarray(normal) / 255.0
    #         normal = (normal - 0.5) * 2.0
    #         normal = torch.from_numpy(normal)
    #         normal.select(-1, 1).neg_()
    #         normal = ((normal + 1.0) / 2.0).clamp(0, 1.0)
    #         normal = (np.asarray(normal) * 255).astype(np.uint8)
    #         cv2.imwrite(path.replace('Normal', 'NormalDX'), normal)
    #         break
    # continue
    
    ## extract text
    keys = dir_name.split('-')[:-1]
    text = ' '.join(keys) 
    text_path = os.path.join(data_dir, dir_name, 'text_wo_label.txt')
    if not os.path.exists(text_path):
        with open(text_path, "w") as f:
            f.write(text)
    
    ## read text
    # if not os.path.exists(text_path):
    #     continue
    with open(text_path, "r") as f:
        text = f.read().strip()
    
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
            
    text = text + (" with {} pattern".format(' and '.join(pattern_list)) if len(pattern_list) > 0 else '')
    text_path = os.path.join(data_dir, dir_name, 'new_text.txt')
    with open(text_path, 'w') as f:
        f.write(text)
    
    # if len(text) == 0:
    #     print(i, ' ', dir_name)#, end='')
    # continue
    
    print(i, ' ', dir_name)#, end='')
    # assert 0
    continue
    l = processing(id_name, dir_name, data_dir, device)
    if l:
        start = time.time()
        out = r.evaluate(*l)

        gt_image = out.permute((0, 2, 3, 1)).cpu().clamp_(0, 1)[0]
        np_gt_image = gt_image.numpy()

        gt_img = (np_gt_image * 255).astype(np.uint8)
        gt_img = Image.fromarray(gt_img)
        gt_img.save(os.path.join(data_dir, dir_name, "render_1K_.png"))
        gt_img_small = gt_img.resize((512, 512))
        gt_img_small.save(os.path.join(data_dir, dir_name, "render_512.png"))
        end = time.time()
        print("\t{:.1f}".format(end - start))
    # assert 0
print(count)
