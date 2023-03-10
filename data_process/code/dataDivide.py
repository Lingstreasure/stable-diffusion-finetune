import os
import sys
import random
import argparse
from shutil import copy

parser = argparse.ArgumentParser(description="Divide the dataset into train and test")
group = parser.add_mutually_exclusive_group()
parser.add_argument("root", type=str, help="The path of the dataset")
parser.add_argument("--names", type=str, nargs=2, help="The names of divided subsets which must contrains "
                                                                  "only two elements")
parser.add_argument("-d", "--dirs", type=str, help="The dirs to store divided dataset e.g. 'train'")
parser.add_argument("-m", "--method", type=str, choices=["random", "bisection"], default="random",
                    help="The method of division")
group.add_argument("-r", "--ratio", type=float, help="The ratio of test_num/data_num")
group.add_argument("-n", "--number", type=int, help="The number of test sample")

arg = parser.parse_args()
data_dir = arg.root
last_key = data_dir.split('/')[-1]
data_pre = data_dir.split('/')[-2] if last_key == '' else last_key

assert len(arg.names) == 2, "Wrong number of subsets"


# 生成划分后子集样本名称文件
def generate_sample_listFile():
    assert os.path.exists(data_dir), "Wrong path!"
    dataFile_list = os.listdir(data_dir)
    dataFile_list.sort()
    sample_num = len(dataFile_list)
    print("sample number is :{}".format(sample_num))

    # 生成子集样本数
    if arg.ratio is not None:
        ratio = arg.ratio
        test_num = int(sample_num * ratio)
        train_num = sample_num - test_num
    elif arg.number is not None:
        assert arg.number < sample_num, "test_num > sample_num"
        test_num = arg.number
        train_num = sample_num - arg.number
    else:
        assert 0, "No number was given"
    print(f"{arg.names[0]}_num:{train_num}, {arg.names[1]}_num:{test_num}")

    sample_name_list = dataFile_list

    # 生成测试集索引列表
    test_index_list, train_index_list = [], []
    if arg.method == "random":
        index_list = [i for i in range(sample_num)]
        random.shuffle(index_list)
        test_index_list = index_list[:test_num]
        train_index_list = index_list[test_num:]
    elif arg.method == "bisection":
        train_index_list = [i for i in range(train_num)]
        test_index_list = [i for i in range(train_num, sample_num)]
    else:
        assert 0, "Wrong divide method!"

    # 生成测试集、训练集样本列表
    test_list, train_list = [], []
    for index in test_index_list:
        test_list.append(sample_name_list[index])
    for index in train_index_list:
        train_list.append(sample_name_list[index])
    assert len(test_list) == test_num, "Wrong test_list_num"

    train_list.sort()
    test_list.sort()
    
    for i, elem in enumerate(train_list):
        train_list[i] = elem + '\n'
    for i, elem in enumerate(test_list):
        test_list[i] = elem + '\n'

    sub_list = [train_list, test_list]

    # 写入文件
    for i, name in enumerate(arg.names):
        file_path = data_pre + '_' + name + ".txt"
        with open(file_path, 'w') as f:
            f.writelines(sub_list[i])
    
        if os.path.exists(file_path):
            print('Make {} successfully!'.format(name + '.txt'))
        else:
            print("{} already exist!".format(name + '.txt'))
            

# 创建目录
def mkdir(path):
    if os.path.exists(path):
        print(path + ' already exist')
    else:
        os.makedirs(path)
        print(path + ' successfully made')
    assert os.path.exists(path), "Directory do not exist"


# 根据列表拷贝文件
def copy_file_with_list(list_dir, input_dir, output_dir, suffix):
    mkdir(output_dir)
    assert os.path.exists(list_dir)
    assert os.path.exists(input_dir)
    assert os.path.exists(output_dir)
    with open(list_dir, 'r') as file:
        name_list = file.read().splitlines()
        name_len = len(name_list)
    # 索引文件
    i = 1
    for name in name_list:
        source = f'{input_dir}/{name}.{suffix}'
        if os.path.exists(source):
            if i == 1:
                print("Start copying!!!")
            try:
                copy(source, output_dir)
            except IOError as e:
                print("Unable to copy file. %s" % e)
                exit(1)
            except:
                print("Unexpected error:", sys.exc_info())
                exit(1)
            i += 1
        if i % 100 == 0:
            print("Already copying %d files" % (i))
    dataFile_list = os.listdir(output_dir)
    target_len = len(dataFile_list)
    if name_len == target_len:
        print("Divide successfully! num:{}".format(name_len))
        return True
    else:
        print("Error!The number of two list do not equal! name_len:{} target_len:{}".format(name_len, target_len))
        return False


if __name__ == "__main__":
    generate_sample_listFile()
    for name in arg.names:
        subset_name_dir = os.getcwd() + '/' + name + '.txt'
        # if arg.dirs is not None:
        #     # 创建子目录
        #     subset_dir = os.getcwd() + '/' + name
        #     mkdir(subset_dir)
        #     # 根据列出的数据集列表进行划分
        #     dir = arg.dirs
        #     in_dir = os.getcwd() + '/' + dir
        #     out_dir = subset_dir + '/' + dir
        #     dirFile_list = os.listdir(in_dir)
        #     suffix = dirFile_list[0].split('.')[-1]
        #     copy_file_with_list(subset_name_dir, in_dir, out_dir, suffix)
        # else:
        #     print("The output dir is not given!!!")
