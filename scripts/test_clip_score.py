import os, sys, argparse
sys.path.append(os.getcwd())
import torch
from torch.utils.data import DataLoader
from ldm.metrics.clip_score import CLIPScore
from ldm.data.simple import Text2MaterialImprove

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--device_num",
        type=int,
        default=0,
        help="The index of gpu for testing clip score" 
    )
    parser.add_argument(
        "--clip_type", 
        type=str,
        default="clip-vit-large-patch14",
        choices=["clip-vit-base-patch16", 
                 "clip-vit-base-patch32", 
                 "clip-vit-large-patch14-336", 
                 "clip-vit-large-patch14"],
        help="The type of clip for testing"
    )
    parser.add_argument(
        "-t",
        "--text_type",
        type=str,
        choices=["full_text", "text_wo_color", "text_wo_pattern", "source_text"], 
        default="full_text", 
        help="The type of text to be tested for clip score."
    )
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="/root/hz/DataSet/mat/data", 
        help="The root dir of all datasets"
    )
    parser.add_argument(
        "--data_list_file_dir",
        type=str, 
        default="./data_file", # relative path of working dir
        help="The dir of all dataset list file"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()
    
    device = "cuda:{}".format(opt.device_num)
    
    print("################ Prepaing Metric: Clip Score ################")
    clip = CLIPScore(f"openai/{opt.clip_type}").to(device)
    clip.requires_grad_(False)
    
    print("################ Prepaing Dataset ################")
    dataset_dict = {}
    for t in ["ambient", "polyhaven", "sharetextures", "3dtextures"]:
        train_dataset = Text2MaterialImprove(data_root_dir=opt.data_root_dir, 
                                            data_list_file_dir=opt.data_list_file_dir, 
                                            dataset_names=[t], 
                                            mode='train', 
                                            )
        test_dataset = Text2MaterialImprove(data_root_dir=opt.data_root_dir, 
                                            data_list_file_dir=opt.data_list_file_dir, 
                                            dataset_names=[t],
                                            mode='test', 
                                            )
        train_dataloader = DataLoader(train_dataset, batch_size=16,num_workers=12, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=16,num_workers=12, shuffle=False)
        dataset_dict[t] = [train_dataloader, test_dataloader]
    
    print("################ Begin Calculating ################")
    data_scores = {}
    for d_type, datas in dataset_dict.items():
        for data in datas:
            length = len(data)
            for i, inputs in enumerate(data):
                images = inputs['image'].to(device)
                text = inputs['txt']
                clip.update(images, text)
                print(f"{i}/{length}")
        
        final_clip_score = clip.compute()
        data_scores[d_type] = final_clip_score
        print(d_type, '\t', final_clip_score)
        clip.clear()
        
    print("################ Final Clip Score ################")
    total_scores = 0
    for k, v in data_scores.items():
        total_scores += v
        print(k, '\t', "{:.2f}".format(v))
    all_score = total_scores / 4
    print("all", '\t', "{:.2f}".format(all_score))
    
    ## 1 new / 2 add "A texture map of" / 3 add "A center flashed texture map of" / 4 wo "arranged in a ?x? grid" / 5 wo "?x?" / 6 "pattern"->"formation"
    ## 7 "?x?" -> "of ? rows and ? columns" + 6 + 8 / 8 "[]" -> "common" / 
    # train , test  1      2      3      4      5      6      7      8      9 
    # full_text   21.92  26.15  26.28  21.92  21.92  21.95  21.90  21.96  
    # wo_pattern  20.82  25.57                      
    # wo_color    21.89  25.93          
    # source      20.52  24.76         
    
    ## full text      1       2         3         4         5         6         7         8 
    # ambient            /  25.95  /  26.19  /  21.37  /  21.45  /  21.47  /  21.41  /  21.48
    # polyhaven          /  25.95  /  25.98  /  20.83  /  20.86  /  20.86  /  20.84  /  20.87
    # 3dtextures         /  26.58  /  26.63  /  22.37  /  22.91  /  22.45  /  22.40  /  22.51
    # sharetextures      /  26.16  /  26.30  /  22.71  /  22.46  /  23.00  /  22.93  /  22.98
    
    ## old / add "A texture map of" / add "Texture map: "
    # train 1990, test 484
    # full_text   21.67  26.05  25.16
    # wo_pattern  20.65  25.48
    # wo_color    21.74  25.89
    # source      20.30  24.72  23.59
    # old complete text: 21.63  25.96