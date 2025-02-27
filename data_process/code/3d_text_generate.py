import os
import os.path as opt


colors = ['blue', 'light', 'dark', 'red', 'green', 'yellow', 'orange', 
          'white', 'black', 'brown', 'off-white', 'beige', 'cyan', 
          'pink', 'purple', 'grey']

data_dir = "/root/hz/DataSet/mat/data/3dtextures"
names = os.listdir(data_dir)
names.sort()
print("Total len: ", len(names))
# count = 0
low_bound, up_bound = 600, 700
for i in range(0, len(names)):
    if i < low_bound:
        continue
    if i >= up_bound:
        break
    
    ### generate text: source text, text_wo_color, text_wo_pattern, full_text
    dir_name = names[i]
    sample_dir = opt.join(data_dir, dir_name)
    text_wo_pattern_path = opt.join(sample_dir, 'text_wo_label.txt')
    full_text_path = opt.join(sample_dir, 'new_text.txt')
    if not opt.exists(text_wo_pattern_path):
        continue
    if not opt.exists(full_text_path):
        continue
    
    ## create texts dir
    text_dir = opt.join(sample_dir, 'texts')
    # os.makedirs(text_dir, exist_ok=True)
    
    ## source text
    # source_keys = dir_name.split('-')[:-1]
    # source_text = ' '.join(source_keys)
    source_text_path = opt.join(text_dir, 'source_text.txt')
    # with open(source_text_path, 'w') as f:
    #     f.write(source_text)
        
    ## text without pattern
    # text_wo_pattern = str()
    # with open(text_wo_pattern_path, 'r') as f:
    #     text_wo_pattern = f.read().strip()
    text_wo_pattern_new_path = opt.join(text_dir, 'text_wo_pattern.txt')
    # os.system(f"cp {text_wo_pattern_path} {text_wo_pattern_new_path}")
    
    ## full text
    # full_text = str()
    # with open(full_text_path, 'r') as f:
    #     full_text = f.read().strip()
    full_text_new_path = opt.join(text_dir, 'full_text.txt')
    # os.system(f"cp {full_text_path} {full_text_new_path}")
    
    ## text without color
    # text_wo_color_keys = []
    # keys = full_text.split(' ')
    # for k in keys:
    #     if k not in colors:
    #         text_wo_color_keys.append(k)          

    # text_wo_color = ' '.join(text_wo_color_keys).strip()
    text_wo_color_path = opt.join(text_dir, 'text_wo_color.txt')
    # with open(text_wo_color_path, 'w') as f:
    #     f.write(text_wo_color)
    
    # count += 1
    print(f"{i} {dir_name} ############")
    
    ## check all text
    source_text = str()
    text_wo_pattern = str()
    text_wo_color = str()
    full_text = str()
    with open(source_text_path, 'r') as f:
        source_text = f.read().strip()
    with open(text_wo_pattern_new_path, 'r') as f:
        text_wo_pattern = f.read().strip()
    with open(text_wo_color_path, 'r') as f:
        text_wo_color = f.read().strip()
    with open(full_text_new_path, 'r') as f:
        full_text = f.read().strip()
    print("source\t", source_text)
    print("-pat\t", text_wo_pattern)
    print("text\t", full_text)
    print("-color\t", text_wo_color)