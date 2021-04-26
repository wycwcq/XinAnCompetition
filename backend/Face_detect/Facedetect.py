from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import sys
import os
import numpy as np
import pandas as pd
import cv2
from skimage import io
# print(os.getcwd())
sys.path.append('./backend')
from Video_handle.Video import Video_Deal


def Face_detect_embedding(image, device):
    """
    该函数的作用是提取imagez中的人脸区域并提取深度特征
    params: image: 通过cv2或者skimage读入的图片
    return: img_aligned: 维度为[n, 3, 160, 160]提取到的人脸区域
            embeddings: 维度为[n, 512], 人脸区域对应的深度特征
    """
    # 模型定义
    '''
    params: image_size为输出图像维度
    '''
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=True,
        device=device
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

     # 检测人脸区域
    img_aligned, porb = mtcnn(image, return_prob=True)
    if img_aligned == torch.Size([]):
        img_aligned = None
    
    embeddings = None
    # 提取深度特征
    if img_aligned != None:
        img_aligned = img_aligned.to(device)
        embeddings = resnet(img_aligned).detach().cpu()

    return img_aligned, embeddings
    
def Compare_Faces(example_embeddings, frame_embeddings, likes_threshold):
    '''
    该函数的作用是比较人脸深度特征之间的相似性
    params: example_embeddings: dict类型， key是人名，value是对应的深度特征
            detected_exp_faces: list类型， 包含一系列embeddings
    return: str类型，检测到人物的名字
    '''
    for name in example_embeddings:
        example = example_embeddings[name]
        for face in frame_embeddings:
            dist = (example - face).norm().item()
            if dist >= likes_threshold:
                return name
    return None


def Detect_ExampleFace(example_faces, device):
    '''
    该函数的作用是对肖像照进行人脸检测并提取特征
    params: example_faces: 字典类型， key是人名，value是对应的url
    return: detected_exp_faces: 字典类型，key是人名，value是对应的embedding
    '''

    # 读取图片
    images = {}
    for name in example_faces:
        img = io.imread(example_faces[name])
        images[name] = img


    # 检测人脸区域并提取深度特征
    images_crob = {}
    detected_exp_faces = {}
    not_detect = 0
    many_faces = 0
    no_faces = 0
    for name in images:
        img_aligned, embeddings = Face_detect_embedding(images[name], device)
        if img_aligned == None:
            not_detect = 1
            print("此张肖像照未检测到人脸!")
        elif img_aligned.size(0) > 1:
            many_faces = 1
            print("请输入只含有一张人脸的肖像照!")
        else:
            images_crob[name] = img_aligned
            detected_exp_faces[name] = embeddings
    if not images_crob:
        no_faces = 1
        print("所有肖像照均未检测到人脸，请重新选择!")
        return None

    return detected_exp_faces

def Detect_VideoFace(video_frames, faces_path, num_threshold, device):
    """
    该函数的作用是将单个video中检测出的人脸的深度特征与肖像照中的深度特征进行比较，
    以确认该视频中是否含有对应的人脸
    params:
    video_frames: list类型，包含图像帧
    detected_exp_faces: 字典类型，name->深度特征
    num_threshold: int类型，表示该视频中存在百分之多少帧包含目标人物时认为该视频包含目标人物
    return:
    bool类型，表示该视频中是否存在目标人物
    """
    num_frames = len(video_frames)
    detected_exp_faces = Detect_ExampleFace(faces_path, device)
    detected_frames = 0
    likes_threshold = 0.5
    frame_i = 0
    for frame in video_frames:
        frame_i += 1
        img_aligned , embeddings = Face_detect_embedding(frame, device)
        if img_aligned != None:
            detect_name = Compare_Faces(detected_exp_faces, embeddings, likes_threshold)
            if detect_name != None:
                print("This frame {} has {}!".format(frame_i, detect_name))
            detected_frames += 1
            if detected_frames >= int(num_threshold * num_frames):
                return True
        else:
            continue

if __name__ == '__main__':
    video_url = 'http://qrshcr4rw.hn-bkt.clouddn.com/video/Obama_short_1.mp4'
    example_faces = {
        'Obama': 'http://qrshcr4rw.hn-bkt.clouddn.com/image/obama.png',
        'Pujing': 'http://qrshcr4rw.hn-bkt.clouddn.com/image/pujing.png'
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    video_deal = Video_Deal(video_url)
    video_frames, _, _, _ = video_deal.video_read()
    if Detect_VideoFace(video_frames=video_frames, faces_path=example_faces, num_threshold=0.3, device=device):
        print("Target characters in the video!")
    else:
        print("Target characters not in the video!")

