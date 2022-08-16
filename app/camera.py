import cv2
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing import image

model = model_from_json(open("model\model.json", "r").read())
model.load_weights('model\model.h5')

print('Model loaded sucessfully')
face_haar_cascade=cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
                
        height, width , channel = frame.shape
        sub_img = frame[0:int(height/6),0:int(width)]

        black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
        res = cv2.addWeighted(sub_img, 0.77, black_rect,0.23, 0)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.8
        FONT_THICKNESS = 2
        lable_color = (10, 10, 255)
        lable = "Emotion Detection"
        lable_dimension = cv2.getTextSize(lable,FONT ,FONT_SCALE,FONT_THICKNESS)[0]
        textX = int((res.shape[1] - lable_dimension[0]) / 2)
        textY = int((res.shape[0] + lable_dimension[1]) / 2)
        cv2.putText(res, lable, (textX,textY), FONT, FONT_SCALE, (0,0,0), FONT_THICKNESS)
        gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_haar_cascade.detectMultiScale(gray_image )
        try:
            for (x,y, w, h) in faces:
                
                cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (0,255,255),thickness =  2)
                
                roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
                roi_gray=cv2.resize(roi_gray,(48,48))

                im = Image.fromarray(roi_gray, 'RGB')
                img_aray = np.array(im)

                image_pixels = np.expand_dims(img_aray, axis=0)
                print(image_pixels.shape)


                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                emotion_prediction = emotion_detection[max_index]
                cv2.putText(res, "Sentiment: {}".format(emotion_prediction), (0,textY+22+5), FONT,0.7, lable_color,2)
                lable_violation = 'Confidence: {}'.format(str(np.round(np.max(predictions[0])*100,1))+ "%")
                violation_text_dimension = cv2.getTextSize(lable_violation,FONT,FONT_SCALE,FONT_THICKNESS )[0]
                violation_x_axis = int(res.shape[1]- violation_text_dimension[0])
                cv2.putText(res, lable_violation, (violation_x_axis,textY+22+5), FONT,0.7, lable_color,2)
        except :
            pass
        frame[0:int(height/6),0:int(width)] =res                
        ret,jpg=cv2.imencode('.jpg',frame)
        
        return jpg.tobytes()

# https://www.datacamp.com/tutorial/face-detection-python-opencv