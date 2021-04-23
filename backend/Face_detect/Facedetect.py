from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import sys
import os
import numpy as np
import pandas as pd
# print(os.getcwd())
sys.path.append('./backend')
from Video_handle.Video import Video_Deal
