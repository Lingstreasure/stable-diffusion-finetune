import json
import os
import requests
import time
# from func_timeout import func_set_timeout


type_lists = ["_nor_dx_4k.jpg", 
              "_rough_4k.jpg", 
              "_diff_4k.jpg", 
              "_disp_4k.jpg", 
              "_ao_4k.jpg", 
              "_metal_4k.jpg", 
              "_spec_4k.jpg",  
              "_col_2_4k.jpg", 
              "_col_1_4k.jpg", 
              "_col_01_4k.jpg", 
              "_col_02_4k.jpg"
              "_col_03_4k.jpg", 
              "_col1_4k.jpg", 
              "_col2_4k.jpg", 
              "_coll1_4k.jpg", 
              "_coll2_4k.jpg"]

# @func_set_timeout(20)
def download_content(url):
    global type_lists
    res_dict = {}
    for file_type in type_lists:
        file_url = url + file_type
        result = requests.get(file_url)
        if result.status_code == 200:
            res_dict[file_type] = result.content
    return res_dict

def save_to_file(filename, content):
	fo = open(filename, "wb")
	fo.write(content)
	fo.close()

#########################################################
### https://polyhaven.com/textures
## stage 1: get all info list
# http_url = "https://api.polyhaven.com/assets?t=textures"
# con = requests.get(http_url).text
# result = json.loads(con)
# with open("polyhaven_textures.json", "w", encoding='utf-8') as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)
# assert 0
## stage 2: get material name and tag
# names_tags = {}
# with open("polyhaven_textures.json", 'r') as f:
#     data_info = json.load(f)

# for name, detailedInfo in data_info.items():
#     names_tags[name] = detailedInfo["tags"]
    
## write tags
# save_dir = "/home/d5/hz/DataSet/mat/polyhaven"
# for k, v in names_tags.items():
#     save_path = os.path.join(save_dir, k) + "/text_wo_label.txt"
#     v = v + k.split('_')[:-1]
#     for i, tag in enumerate(v):
#         v[i] = tag + ' '
#     with open(save_path, 'w') as f:
#         f.writelines(v)
#     print(k)
# assert 0
### download
# progress = 233
# while True:
#     try:
#         url_format = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/{}/{}"
#         save_dir = "/home/d5/hz/DataSet/mat/polyhaven"
#         for i, name in enumerate(names_tags.keys()):
#             if i < progress:
#                 continue
#             start = time.time()
#             down_url_prefix = url_format.format(name, name)
#             res_dict = download_content(down_url_prefix)
#             file_dir = os.path.join(save_dir, name)
#             os.makedirs(file_dir, exist_ok=True)
#             for file_name, file_content in res_dict.items():
#                 file_path = os.path.join(file_dir, file_name)
#                 save_to_file(file_path, file_content)
            
#             print(i + 1, '\t', name, '\t', time.time() - start)
#             progress = i + 1
#     except BaseException as e:
#         print(e)
#     finally:
#         break

import cv2
import numpy as np
path = "/home/d5/hz/DataSet/mat/polyhaven/brown_brick_02"
diff_path = os.path.join(path, "_diff_4k.jpg")
spec_path = os.path.join(path, "_spec_4k.jpg")
diffuse = np.array(cv2.imread(diff_path)) / 255.0 
specular = np.array(cv2.imread(spec_path)) / 255.0
basecolor = (np.clip((diffuse + specular) * 255.0, 0, 255 )).astype(np.uint8)
print(np.max(basecolor), np.min(basecolor))
cv2.imwrite("/home/d5/hz/DataSet/mat/polyhaven/brown_brick_02/_color_4k_.png", basecolor)