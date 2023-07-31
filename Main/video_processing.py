from fer import FER
from fer import Video
import pandas as pd
video_file: str= "./Videos/v1.mp4"
face_detector: FER = FER(mtcnn=True)
#preprocess video file 
processed_video = Video(video_file=video_file)
prcessing_data = processed_video.analyze(face_detector,display=True)