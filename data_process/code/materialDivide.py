from math import floor
import os#, sh, random
# from curses.ascii import isdigit

if __name__ == '__main__':
    # data_dir = "/home/d5/hz/DataSet/material"
    # sample_folder_names = sorted(os.listdir(data_dir))
    ### divide material dataset
    # bias_of_category = [0]
    # len_of_category = list()
    # bias = 0
    # category = 'A'
    # for folder_name in sample_folder_names:
    #     cur_category = folder_name.split('_')[0][0]
    #     if cur_category != category:
    #         category = cur_category
    #         len_of_category.append(bias - bias_of_category[-1])
    #         bias_of_category.append(bias)
    #     bias += 1
    # len_of_category.append(bias - bias_of_category[-1])  # last category
        
    # data_len = len(sample_folder_names)
    # train_sample_folder_idx, test_sample_folder_idx = [], []
    # for i in range(len(bias_of_category)):
    #     cur_caty_bias = bias_of_category[i]
    #     target_idx = floor(random.uniform(cur_caty_bias, cur_caty_bias + len_of_category[i]))
    #     test_sample_folder_idx.append(target_idx)
    # for idx in range(data_len):
    #     if idx not in test_sample_folder_idx:
    #         train_sample_folder_idx.append(idx)
        
    # sh.mkdir("material/train")
    # sh.mkdir("material/test")
    # print("Total nums : {}".format(data_len))
    # count = 0
    # for name_idx in test_sample_folder_idx:
    #     # sh.mv(os.path.join(data_dir, sample_folder_names[name_idx]), "test")
    #     os.system("mv " + os.path.join(data_dir, sample_folder_names[name_idx]) + ' ' + os.path.join(data_dir, "test"))
    #     count += 1
    #     print("finished {}".format(count))
    # for name_idx in train_sample_folder_idx:
    #     os.system("mv " + os.path.join(data_dir, sample_folder_names[name_idx]) + ' ' + os.path.join(data_dir, "train"))
    #     count += 1
    #     print("finished {}".format(count))  
    
    # with open("./tag.txt", 'w', encoding='utf-8') as f:
    #     for name in sample_folder_names:
    #         f.writelines(name+'\n')
    
    ### extract text
    # for i, sample_name in enumerate(sample_folder_names):
    #     raw_text = sample_name
    #     keys = raw_text.split('_')
    #     core_keys = str()
    #     ## remove digits
    #     for ch in keys[0]:
    #         if not isdigit(ch):
    #             core_keys += ch
                
    #     ## split key
    #     core_keys_split = str()
    #     core_key = str()
    #     for ch in core_keys:
    #         if ch.isupper():
    #             core_keys_split += core_key
    #             core_keys_split +=  ' ' if len(core_key) != 0 else ''
    #             core_key = str()
    #         core_key += ch

    #     if len(core_key) == 1:
    #         if core_key == 'L':
    #             core_keys_split = 'large ' + core_keys_split[:-1]
    #         elif core_key == 'S':
    #             core_keys_split = 'small ' + core_keys_split[:-1]
    #     else:    
    #         core_keys_split += core_key

    #     core_keys_split = core_keys_split.lower()
    #     str_checker = core_keys_split.split(' ')
    #     filter_keys = []
    #     for key in keys[1:]:
    #         is_repeat = False
    #         for k in str_checker:
    #             if key.startswith(k) or k.startswith(key):
    #                 is_repeat = True
    #                 break
    #         if not is_repeat:
    #             filter_keys.append(key)
        
    #     filtered_key_idx = []
    #     for p in range(len(filter_keys)):
    #         chWord = filter_keys[p]
    #         for q in range(len(filter_keys)):
    #             if q == p:
    #                 continue
    #             if filter_keys[q].startswith(chWord):
    #                 filtered_key_idx.append(p)
        
    #     more_filter_keys = []
    #     for j in range(len(filter_keys)):
    #         if j in filtered_key_idx:
    #             continue
    #         else:
    #             more_filter_keys.append(filter_keys[j])
        
    #     text = ' '.join(more_filter_keys) + " {}".format(core_keys_split)
    #     ## texture pattern 
    #     # label_path = os.path.join(data_dir, sample_name, 'label.txt')
    #     # if not os.path.isfile(label_path):
    #     #     continue
    #     # with open(label_path, 'r') as label_f:
    #     #     label = label_f.readline()
    #     #     label = label.strip()
    #     #text = text + " with {} pattern".format(label)
        
    #     text_path = os.path.join(data_dir, sample_name, 'text_wo_label.txt')
    #     with open(text_path, 'w') as text_f:
    #         text_f.write(text)
    #     print("finished {}".format(i + 1))
    
    ### transfer txt
    # tar_data_dir = "/home/d5/hz/DataSet/mat/data"
    # tar_sample_folder_names = os.listdir(tar_data_dir)
    # count = 1
    # for tar_folder_name in tar_sample_folder_names:
    #     for folder_name in sample_folder_names:
    #         if folder_name.startswith(tar_folder_name):
    #             sour_txt_path = os.path.join(data_dir, folder_name, "text_wo_label.txt")
    #             tar_txt_dir = os.path.join(tar_data_dir, tar_folder_name)
    #             os.system(f"cp {sour_txt_path} {tar_txt_dir}")
    #             print(f"finished {count}")
    #             count += 1
    
    ### see the model ckpt content
    # import torch
    # model_ckpt_path = "/home/d5/hz/Code/stable-diffusion-finetune/models/ldm/512-base-ema.ckpt"
    # model_ckpt_path1 = "/home/d5/hz/Code/stable-diffusion-finetune/models/ldm/sd-v1-4-full-ema.ckpt"
    # model_param = torch.load(model_ckpt_path)
    # model_param1 = torch.load(model_ckpt_path1)
    # sd = model_param['state_dict']
    # sd1 = model_param1['state_dict']
    # count = 0
    # for k in sd1.keys():
    #     # if k not in sd:
    #     if k.startswith("first") or k.startswith("cond"):
    #         continue
    #     if k.startswith("model."):
    #         k = k.replace('.', '').replace("modeldiffusion", "model_ema.diffusion")
    #         if k in sd1:
    #             count += 1
    # print(count)
    # print(sd.keys())
    # assert 0
    # keys = []
    # for k, v in sd.items():
    #     if k.startswith('model.'):
    #         count += 1
    #         keys.append(k)
    #         # sd[k.replace('model.', 'model_ema.')] = v

    # for k in keys:
    #     sd[k.replace('.', '').replace("modeldiffusion", "model_ema.diffusion")] = sd[k].detach().clone()
        
    # torch.save(model_param, model_ckpt_path.replace("512-base-ema", "512_with_ema"))
        
    data_dir = "/media/d5/7D1922F98D178B12/hz/DataSet/mat/data"
    list_file_dir = "/home/d5/hz/DataSet/mat/code"
    datasets = ['ambient', '3dtextures', 'polyhaven', 'sharetextures']
    target_file_path = os.path.join(list_file_dir, 'new_test_only_label.txt')
    for data_name in datasets:
        dataset_dir = os.path.join(data_dir, data_name)
        dataset_file_path = os.path.join(list_file_dir, data_name + "_test.txt")
        with open(dataset_file_path, 'r') as f:
            sample_names = f.readlines()
        
        text_list = []
        for name in sample_names:
            name = name.strip()
            txt_path = os.path.join(dataset_dir, name, 'new_text.txt')
            if not os.path.exists(txt_path):
                continue
            txt_path = os.path.join(dataset_dir, name, 'text_wo_label.txt')
            if not os.path.exists(txt_path):
                continue
            with open(txt_path, 'r') as txt_f:
                text = txt_f.read().strip()
            text = text.split(' ')[-1]
            text_list.append(text + '\n')
        with open(target_file_path, 'a') as f:
            f.writelines(text_list)
       
        
    
    
    