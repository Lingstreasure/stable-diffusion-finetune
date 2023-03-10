import json
import os
import requests, gdown
import time
from func_timeout import func_set_timeout

def get_tag_list(text):
    out = []
    start = text.find("3dtextures.me")
    while start > -1:
        text = text[start:]
        end = text.find("/\">")
        if end > -1:
            tag_data = text[:end]
            # print(tag_data)
            text = text[end + 1:]
            start = text.find("3dtextures.me")
            print(tag_data)
            out.append(tag_data.replace("3dtextures.me", ''))
        else:
            break
    # print(out)
    return out

def get_data_list(tag_url):
    out = []
    tag_con = requests.get(tag_url)
    text = tag_con.text
    start = text.find("<article")
    while start > -1:
        text = text[start:]
        end = text.find("class=\"thumb vertical\">")
        if end > -1:
            data_text = text[:end]  # "<article ... class ...>"
            data_start = data_text.find("https:")
            if data_start > -1:
                data_text = data_text[data_start:]
                data = data_text.split('"')[0]
                out.append(data)
            text = text[end:]
            start = text.find("<article")
        else:
            break
    print(len(out))
    return out
    

@func_set_timeout(30)
def download_google_drive_content(url, out_dir):
    gdown.download_folder(url, quiet=True, output=out_dir)#, use_cookies=False)

def download_meocloud_content(url, out_dir):
    file_names = ["_COLOR", "_DISP", "_NRM", 
                  "_OCC", "_SPEC", "_RENDER", 
                  "_ROUGH", "_NORM", "_AO", 
                  ]
    file_types = [".jpg", ".png"]
    os.makedirs(out_dir, exist_ok=True)
    file_name_pre_list = out_dir.split('/')[-1].split('-')
    for i, word in enumerate(file_name_pre_list):  # upper the first ch of word
        if word[0].isalpha():
            file_name_pre_list[i] = word[0].upper() + word[1:]
    file_pre = '_'.join(file_name_pre_list)
    for f_name in file_names:
        for f_type in file_types:
            file_name = file_pre + f_name + f_type
            res = requests.get(url + file_name)
            if res.status_code == 200:
                print(file_name)
                save_to_file(os.path.join(out_dir, file_name), res.content)

def save_to_file(filename, content):
	fo = open(filename, "wb")
	fo.write(content)
	fo.close()

def find_next_page_urls(tag_url: str, urls: str):
    tag_con = requests.get(tag_url)
    text = tag_con.text
    next_page_url_start = text.find('class="nav-previous"')
    if next_page_url_start > -1:
        next_page_data = text[next_page_url_start:]
        next_page_url_end = next_page_data.find("><span")
        next_page_data = next_page_data[:next_page_url_end]
        next_page_url = next_page_data.split('"')[-2]
        urls += (next_page_url+',')  # split by ','
        urls += find_next_page_urls(next_page_url, urls)
        
    return urls

def find_download_url(url_text: str):
    infos = {
             "https://drive.google.com/": 
                 {"pre": "https://drive.google.com/drive/folders/", 
                  "name": "id", 
                  "name_bias": 3, 
                  "type": "google", }, 
             "https://meocloud.pt/": 
                 {"pre": "https://cld.pt/dl/download/", 
                  "name": "link", 
                  "name_bias": 5, 
                  "type": "meocloud", }
            }  # google drive, meocloud
    for url in infos.keys():
        download_url_start = url_text.find(url)  
        if download_url_start != -1:
            download_url_text = url_text[download_url_start:]
            download_url_end = download_url_text.find('" target')
            download_url = download_url_text[:download_url_end]
            name_start = download_url.find(infos[url]["name"])  # name means id
            if download_url.find("folder") > -1:
                return {"url": download_url, "type": infos[url]["type"]}
            elif name_start > -1:
                download_url = infos[url]["pre"] + download_url[name_start + infos[url]["name_bias"]:].split("?")[0]  # extract the name(id)
                return {"url": download_url, "type": infos[url]["type"]}
            else:
                print("Can't find download id")
    return None

##########################################
### https://3dtextures.me/
http_url = "https://3dtextures.me/tag"
## stage 1  get tag 
# con = requests.get(http_url)
# result = con.text
# result = result.encode("GBK", "ignore")
# with open("3dtextures_temp.xml", "wb") as f:
#     f.write(result)
# assert 0

## stage 2  get data url
# text = open("3dtextures_temp.xml", "r").read()
# all_tags = get_tag_list(text)
# url = "https://3dtextures.me"
# data_urls = []
# all_tags_next_page_urls = set()
# for tag in all_tags:
#     # if tag != "/tag/concrete":
#     #     continue
#     tag_url = url + tag
#     for next_page_url in find_next_page_urls(tag_url, str()).split(',')[:-1]:
#         all_tags_next_page_urls.add(next_page_url)
#     data_urls += get_data_list(tag_url)

# for tag_url in all_tags_next_page_urls:
#     data_urls += get_data_list(tag_url)

# for i, url in enumerate(data_urls):
#     data_urls[i] = data_urls[i] + '\n'
    
# with open("3dtextures_data_urls.txt", 'w') as f:
#     f.writelines(data_urls)

## stage 3 get data
data_urls = open("3dtextures_data_urls.txt", 'r').readlines() 
data_dir = "../3dtextures"
os.makedirs(data_dir, exist_ok=True)
already_downloads = os.listdir(data_dir)
already_downloads.sort()
## delete empty dir
# save_dir = "/home/d5/hz/DataSet/mat/3dtextures"
# exist_names = os.listdir(save_dir)
# for name in exist_names:
#     path = os.path.join(save_dir, name)
#     elements = os.listdir(path)
#     if len(elements) < 3:
#         print(path)
#         os.system("rm -r {}".format(path))
# assert 0
progress = 0
while True:
    try:
        funcs = {'goolge': download_google_drive_content, 
                 'meocloud': download_meocloud_content}
        for i, url in enumerate(data_urls):
            url = url.strip()
            folder_name = url.split('/')[-2]
            if i < progress or folder_name in already_downloads:
                continue
            print(url)
            url_text = requests.get(url).text
            # with open(f"{folder_name}.txt", 'wb') as f:
            #     f.write(url_text.encode("GBK", "ignore"))
            # assert 0
            download_info = find_download_url(url_text)
            if not download_info or download_info["type"] == 'google':
                continue
            start = time.time()
            print("download url : ", download_info['url'])
            funcs[download_info['type']](download_info['url'], data_dir + '/' + folder_name)
            print(i + 1, '\t', folder_name, '\t', time.time() - start)
            progress = i + 1
    except BaseException as e:
        print(e)
    finally:
        print("finished!")
        break
    