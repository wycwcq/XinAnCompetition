import cv2
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw

class Video_Deal:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
    
    def video_read(self):
        all_frames = []
        i = 0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame)
                print("{} frame is loaded".format(i))
            else:
                break
            i += 1
        if all_frames:
            print("Video loaded successfully")
            return all_frames, width, height, fps
        else:
            print("Video loading failed")
            return None

if __name__ == '__main__':
    video_url = 'http://qrshcr4rw.hn-bkt.clouddn.com/video/video.mp4'
    video_deal = Video_Deal(video_path=video_url)
    frames, w, h, fps = video_deal.video_read()
    for idx, img in enumerate(frames):
        cv2.imwrite('./backend/Video_handle/video_frames/' + str(idx) + '.png', img)
    # print(frames)
