import torch
from torch.utils.model_zoo import load_url
import matplotlib.pyplot as plt
from scipy.special import expit
import numpy as np

import sys
sys.path.append('./backend/Deepfake_detect2')

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet,weights
from isplutils import utils

def detect_deepfake_video(video):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    net_model = 'Xception'
    train_db = 'DFDC'
    face_policy = 'scale'
    face_size = 224
    frames_per_video = 32

    model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
    net = getattr(fornet,net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))

    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

    facedet = BlazeFace().to(device)
    facedet.load_weights("./backend/Deepfake_detect2/blazeface/blazeface.pth")
    facedet.load_anchors("./backend/Deepfake_detect2/blazeface/anchors.npy")
    videoreader = VideoReader(verbose=False)
    video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)

    video_score = 0

    # 一个视频一个视频进行处理
    # for video in video_urls:
    #     video_name = video.split('/')[-1][:-4]
    #     vid_faces = face_extractor.process_video(video)
        
    #     score = []
    #     # num_frame_has_faces = 0
    #     for frame in vid_faces:
    #         # 如果存在人脸
    #         if len(frame['faces']):
    #             faces = []
    #             for face in frame['faces']:
    #                 faces.append(transf(image=face)['image'])
    #             faces = torch.stack(faces)
    #             faces_pred = net(faces.to(device)).cpu().detach().numpy().flatten()
    #             frame_score = expit(faces_pred.mean())
    #             score.append(frame_score)

    #     if score:
    #         video_score[video_name] = np.mean(score)
    #     else:
    #         video_score[video_name] = 0
    
    video_name = video.split('/')[-1][:-4]
    vid_faces = face_extractor.process_video(video)
    
    score = []
    # num_frame_has_faces = 0
    for frame in vid_faces:
        # 如果存在人脸
        if len(frame['faces']):
            faces = []
            for face in frame['faces']:
                faces.append(transf(image=face)['image'])
            faces = torch.stack(faces)
            faces_pred = net(faces.to(device)).cpu().detach().numpy().flatten()
            frame_score = expit(faces_pred.mean())
            score.append(frame_score)

    if score:
        video_score = np.mean(score)
    else:
        video_score = 0

    return video_score

if __name__ == '__main__':
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    video_urls = 'http://qrshcr4rw.hn-bkt.clouddn.com/video/Obama_short_1.mp4'
    # 这个final_score是该视频为假的概率
    final_score = detect_deepfake_video(video=video_urls)
    print(final_score)
            
            


