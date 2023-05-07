import os
import os.path as opt


colors = ['blue', 'light', 'dark', 'red', 'green', 'yellow', 'orange', 
          'white', 'black', 'brown', 'off-white', 'beige', 'cyan', 
          'pink', 'purple', 'grey']

data_dir = "/root/hz/DataSet/mat/data/polyhaven"
names = os.listdir(data_dir)
names.sort()
print("Total len: ", len(names))
print("Total len: ", len(names))
for i in range(0, len(names)):
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
    os.makedirs(text_dir, exist_ok=True)
    
    ## source text
    source_keys = dir_name.split('_')
    filtered_keys = []
    for j, k in enumerate(source_keys):
        if k[0].isalpha():
            filtered_keys.append(source_keys[j])
            
    source_text = ' '.join(filtered_keys)
    source_text_path = opt.join(text_dir, 'source_text.txt')
    with open(source_text_path, 'w') as f:
        f.write(source_text)
        
    ## text without pattern
    # text_wo_pattern = str()
    # with open(text_wo_pattern_path, 'r') as f:
    #     text_wo_pattern = f.read().strip()
    # text_wo_pattern_new_path = opt.join(text_dir, 'text_wo_pattern.txt')
    # os.system(f"cp {text_wo_pattern_path} {text_wo_pattern_new_path}")
    
    ## full text
    # full_text = str()
    # with open(full_text_path, 'r') as f:
    #     full_text = f.read().strip()
    # full_text_new_path = opt.join(text_dir, 'full_text.txt')
    # os.system(f"cp {full_text_path} {full_text_new_path}")
    
    ## text without color
    # text_wo_color_keys = []
    # keys = full_text.split(' ')
    # for k in keys:
    #     if k not in colors:
    #         text_wo_color_keys.append(k)          

    # text_wo_color = ' '.join(text_wo_color_keys).strip()
    # text_wo_color_path = opt.join(text_dir, 'text_wo_color.txt')
    # with open(text_wo_color_path, 'w') as f:
    #     f.write(text_wo_color)
    
    print(f"{i} {dir_name}")
    # assert 0