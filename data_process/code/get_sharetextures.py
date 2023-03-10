import json
import os
import requests
import time
from func_timeout import func_set_timeout


def get_name_list(text):
    out = []
    start = text.find("title")
    while start > -1:
        text = text[start:]
        name_start = text.find("slug")
        if name_start == -1:
            break
        text = text[name_start + 7:]
        name_end = text.find('"')
        if name_end == -1:
            break
        name = text[: name_end]
        text = text[name_end + 1:]
        cat_start = text.find("category")  # category -> slug1 -> slug2 
        if cat_start == -1:
            break
        text = text[cat_start:]
        slug1_start = text.find("slug")
        if slug1_start == -1:
            break
        text = text[slug1_start + 4:]
        slug2_start = text.find("slug")
        if slug2_start == -1:
            break
        text = text[slug2_start + 7:]
        slug2_end = text.find('"')
        if slug2_end == -1:
            break
        category = text[: slug2_end]
        category_name = category + '/' + name
        out.append(category_name)
        print(category_name)
        text = text[slug2_end:]
        start = text.find("title")

    print(len(out))
    return out

def find_download_url(text, name):
    url_format = "https://files.sharetextures.com/file/"
    start = text.find(url_format)
    if start == -1:
        return None
    text = text[start + 37:]
    end = text.find('/')
    key = text[: end]
    return url_format + key + '/' + name

# @func_set_timeout(20)
def download_content(url):
    results = requests.get(url)
    if results.status_code == 200:
        return results.content
    else:
        print("failed download {}".format(url.split('/')[-1]))
        return None
    
def save_to_file(filename, content):
	fo = open(filename, "wb")
	fo.write(content)
	fo.close()

#########################################################
### https://www.sharetextures.com/
## stage 1: get all info list
# url_format = "https://api2.sharetextures.com/api/v0/for-frontend/items?itemType=textures&sortBy=most_recent&page={}&_u=538902edf66b0d8a7988278e6e244130"
# for i in range(1, 200):
#     http_url = url_format.format(i)
#     info = requests.get(http_url).text.encode("GBK", "ignore")
#     with open("page1.txt", 'wb') as f:
#         f.write(info)

# url = "https://www.sharetextures.com/_next/data/QZsvRxSAsC6saqBdXyYp0/textures/fabric/fabric_123.json"
# res = requests.get(url).text.encode("GBK", "ignore")
# with open("fabric_url.txt", "wb") as f:
#     f.write(res)
# assert 0
## stage 2: get material name
# with open("page1.txt", 'r') as f:
#     data_info = f.read()

# names = get_name_list(data_info)
# for i, name in enumerate(names):
#     names[i] = name + '\n'
    
# with open("sharetextures_name.txt", 'w') as f:
#     f.writelines(names)

## stage 3: download
with open("sharetextures_name.txt", 'r') as f:
    names = f.readlines()
progress = 20
while True:
    try:
        url_format = "https://www.sharetextures.com/textures/{}"
        save_dir = "/home/d5/hz/DataSet/mat/sharetextures"
        for i, name in enumerate(names):
            if i < progress:
                continue
            name = name.strip()
            file_name = name.split('/')[-1].replace('-', '_') + "-4K.zip"
            url = url_format.format(name)
            res = requests.get(url).text
            download_url = find_download_url(res, file_name)
            print(download_url)
            start = time.time()
            result = download_content(download_url)
            file_path = save_dir + '/' + file_name
            save_to_file(file_path, result)
            os.system("unzip '{}' -d {}".format(file_path, save_dir))
            print(i + 1, '\t', name, '\t', time.time() - start)
            progress = i + 1
    except BaseException as e:
        print(e)
    finally:
        break
