import os

if __name__ == '__main__':
    data_dir = "/root/hz/DataSet/mat/data"
    list_file_dir = "/root/hz/Code/stable-diffusion-finetune/data_file"
    datasets = ['ambient', '3dtextures', 'polyhaven', 'sharetextures']
    target_file_path = os.path.join(list_file_dir, 'test_full_text.txt')
    text_list = []
    for data_name in datasets:
        dataset_dir = os.path.join(data_dir, data_name)
        dataset_file_path = os.path.join(list_file_dir, data_name + "_test.txt")
        with open(dataset_file_path, 'r') as f:
            sample_names = f.readlines()
        
        for name in sample_names:
            name = name.strip()
            txt_path = os.path.join(dataset_dir, name, 'texts', 'full_text.txt')
            if not os.path.exists(txt_path):
                continue
            # txt_path = os.path.join(dataset_dir, name, 'texts', 'text_wo_label.txt')
            # if not os.path.exists(txt_path):
            #     continue
            with open(txt_path, 'r') as txt_f:
                text = txt_f.read().strip()
            # text = text.split(' ')[-1]
            text_list.append(text + '\n')
    text_list.sort()
    with open(target_file_path, "w") as f:
        f.writelines(text_list)
    