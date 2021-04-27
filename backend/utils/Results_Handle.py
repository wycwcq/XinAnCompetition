import os
import sys 
import json


def Save_result(User, video_name, key, value, return_path=False):
    '''
    Describe: 该函数的作用对应结果保存到对应的json文件中。
              默认顶层路径为/home/cv/wu/wyc/XinAn/Result_Save, 该函数会将用户名和视频名称连接
              在顶层路径之后, 即'/home/cv/wu/wyc/XinAn/Result_Save' + '/' + User + video_name + '.json'
    params: User, str类型, 表示用户名称; video_name, str类型, 表示视频名称; key, str类型, 表示
            在json文件中的key值; value, obj类型, 表示key对应的value; return_path, bool类型, 表示
            是否返回保存成功的路径
    return: 若保存成功则返回true与path，保存失败则返回false
    '''
    TOP_PATH = '/home/cv/wu/wyc/XinAn/Result_Save' + os.sep
    usr_path = TOP_PATH + User
    if not os.path.exists(usr_path):
        os.mkdir(usr_path)
    save_path = usr_path + os.sep + video_name + '.json'
    if os.path.exists(save_path):
        fr = open(save_path, encoding='utf-8')
        ori_data = json.load(fr)
        fr.close()
        if key not in ori_data:
            ori_data[key] = value
            print("Json updated down")
        else:
            print("The key already exists")
            ori_data[key] = value
        with open(save_path, 'w', encoding='utf-8') as res_file:
            json.dump(ori_data, res_file)
            if return_path:
                return True, save_path
            else:
                return True
    else:
        with open(save_path, 'w', encoding='utf-8') as res_file:
            saved_data = {}
            saved_data[key] = value
            # saved_data = json.dumps(saved_data)
            json.dump(saved_data, res_file)
            print("Json save done!")
            if return_path:
                return True, save_path
            else:
                return True


if __name__ == '__main__':
    User = 'test_user'
    video_name = 'ch'
    # key = 'old'
    key = 'new2'
    # value = ['接下来', '由', '我', '为', '各位', '进行']
    value = None
    Save_result(User, video_name, key, value, return_path=True)


