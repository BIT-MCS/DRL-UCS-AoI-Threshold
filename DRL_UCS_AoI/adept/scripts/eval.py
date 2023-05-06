import os 

def print_list_dir(dir_path):
    dir_list =[]
    dir_files=os.listdir(dir_path) #得到该文件夹下所有的文件
    for file in dir_files:
        file_path=os.path.join(dir_path,file)  #路径拼接成绝对路径
        if os.path.isdir(file_path):  #如果目录，就递归子目录
            #print(file_path)
            dir_list.append(file_path)
    return dir_list
            
if __name__ == '__main__':

    path = '/home/liuchi/wh/logs/EnvBJ-v0'
    all = print_list_dir(path)
    for dir in all:
        print("python evaluate.py --logdir {}".format(dir))
        os.system("python evaluate.py --logdir {}".format(dir))
