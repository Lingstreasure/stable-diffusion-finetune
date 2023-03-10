import json
import os
import requests
import time
from func_timeout import func_set_timeout

def get_material_list(text):
    out = []
    start = text.find("<a href=")
    while start > -1:
        text = text[start:]
        end = text.find(">")
        if end > -1:
            data = text[:end+1]
            # print(data)
            text = text[end+1:]
            start = text.find("<a href=")

            if "<a href=\"/view?id=" in data:
                out.append(data[len("<a href=\"/view?id="):-2])
        else:
            break
    # print(out)
    return out

@func_set_timeout(20)
def download_content(url):
    results = requests.get(url)
    return results.content

def save_to_file(filename, content):
	fo = open(filename, "wb")
	fo.write(content)
	fo.close()

#########################################################
### https://ambientcg.com/list?sort=Popular&offset=0&limit=1800
page_start = 0
limit = 180
http_url = "https://ambientcg.com/list?sort=Popular&offset=%d&limit=%d"

# downloading https://ambientcg.com/get?file=%s_1K-JPG.zip
### record file names to be downloaded
all_data = []
for i in range(10):
    now_url = http_url % (page_start +i * limit, limit)
    con = requests.get(now_url)
    result = con.text
    result = result.encode("GBK", "ignore")
    with open("temp.xml", "wb") as f:
        f.write(result)

    text = open("temp.xml", "r").read()
    all_data += get_material_list(text)
    print(i + 1)

print(len(all_data))
with open("all_material.json", "w") as f:
    f.write(json.dumps(all_data, indent=4))
    
### download
progress = 292
while True:
    try:
        url_format = "https://ambientcg.com/get?file={}_4K-JPG.zip"
        save_dir = "D:/DataSet/mat/data"
        for i, name in enumerate(all_data):
            if i < progress:
                continue
            start = time.time()
            down_url = url_format.format(name)
            result = download_content(down_url)
            file_path = os.path.join(save_dir, name+'.zip')
            save_to_file(file_path, result)
            file_dir = os.path.join(save_dir, name)
            os.makedirs(file_dir, exist_ok=True)
            os.system("unzip '{}' -d {}".format(file_path, file_dir))
            print(i + 1, '\t', name, '\t', time.time() - start)
            progress = i + 1
    except BaseException as e:
        print(e)
    finally:
        break

os.system("rm D:/DataSet/mat/data/*.zip")
    