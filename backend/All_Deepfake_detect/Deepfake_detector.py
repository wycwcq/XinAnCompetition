import os 
import sys
sys.path.append('./backend/Deepfake_detect1')
sys.path.append('./backend/Deepfake_detect2')
# from Deepfake_detect1.detect_from_video import test_full_image_network
# from Deepfake_detect1.test_url_video import detect_deepfake_video
from detect_from_video import test_full_image_network
from test_url_video import detect_deepfake_video


def detect_deepfake_video_all(video_url):
    score1 = test_full_image_network(video_path=video_url)
    print("The result of the first algorithm is {}".format(score1))
    score2 = detect_deepfake_video(video_url)
    print("The result of the second algorithm is {}".format(score2))
    final_score = (score1 + score2) / 2
    return final_score
    # eturn score2

if __name__ == '__main__':
    video_path = 'http://qrshcr4rw.hn-bkt.clouddn.com/video/Obama_short_1.mp4'
    score = detect_deepfake_video_all(video_path)
    print(score)