from fer import FER
import cv2

emotion_detector = FER(mtcnn= True)

img = cv2.imread("./Images/p1.jpg")
img_resized = cv2.resize(img, (600,600))


emotions = emotion_detector.detect_emotions(img_resized)
print(emotions)

top_emotion, emotion_score = emotion_detector.top_emotion(img_resized)
print(top_emotion,emotion_score)

cv2.imshow("Image with emotions", img_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()