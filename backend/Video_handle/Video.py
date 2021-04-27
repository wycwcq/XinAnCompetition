import cv2
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
import os
from moviepy.editor import *

class Video_Deal:
    def __init__(self, video_paths):
        '''
        params: video_paths, list类型, 每一个元素为一个视频路径
        '''
        self.video_path = video_paths
        self.video_name = []
        for video in self.video_path:
            self.video_name.append(video.split('/')[-1][:-4])
        
    def video_read(self):
        '''
        return: video_frames, dict类型, key为视频名称, value为视频信息
        '''
        video_frames = {}
        for video in self.video_path:
            video_name = video.split('/')[-1][:-4]
            cap = cv2.VideoCapture(video)    
            all_frames = []
            i = 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    all_frames.append(frame)
                    print("{} video {} frame is loaded".format(video_name, i))
                else:
                    break
                i += 1
            if all_frames:
                print("Video {} loaded successfully".format(video_name))
                # return all_frames, width, height, fps
                video_frames[video_name] = [all_frames, width, height, fps]
            else:
                print("Video loading failed")
                # return None
                video_frames[video_name] = None

        return video_frames

    def Split_Audio(self, audio_save_path, audio_save_format):
        '''
        Describe: 该函数的作用是将视频中的音频拆分出来，并将音频保存到本地(后期可以上传到七牛云)
        params: audio_save_path, str类型, 表示视频保存顶层路径
        return: 若分离成功则返回对应系统路径(格式为: audio_save_path+name+.mp3)，若分离失败则返回None
        '''
        video_audios = {}
        for video in self.video_path:
            video_name = video.split('/')[-1][:-4]
            audio_final_save_path = audio_save_path + os.sep + video_name + '.' + audio_save_format
            if os.path.exists(audio_final_save_path):
                print("The {} video's audio already exists!".format(video_name))
                continue
            now_video = VideoFileClip(video)
            audio = now_video.audio
            audio.write_audiofile(audio_final_save_path)
            video_audios[video_name] = audio_final_save_path
        return video_audios




if __name__ == '__main__':
    video_url1 = 'http://qrshcr4rw.hn-bkt.clouddn.com/video/video.mp4'
    video_url2 = 'http://qrshcr4rw.hn-bkt.clouddn.com/video/Obama_short_1.mp4'
    video_url3 = 'http://qrshcr4rw.hn-bkt.clouddn.com/video/Chinese_short.mp4'
    video_url = [video_url1, video_url2, video_url3]
    video_deal = Video_Deal(video_paths=video_url)
    video_name = video_deal.video_name
    # video_frames = video_deal.video_read()
    video_audios = video_deal.Split_Audio(audio_save_path='/home/cv/wu/wyc/XinAn/test/audios', audio_save_format='mp3')
    # print(video_audios)
    # frames, w, h, fps = video_deal.video_read()
    # for idx, img in enumerate(frames):
    #     cv2.imwrite('./backend/Video_handle/video_frames/' + str(idx) + '.png', img)
    # print(frames)
