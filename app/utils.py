import cv2
import numpy as np
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing import image


# load model

model = model_from_json(open("model\model.json", "r").read())
model.load_weights('model\model.h5')
print('Model loaded sucessfully')


def pipeline_model(path,filename):
    img_path=path
    
    test_image=image.load_img(img_path,target_size=(48,48))
    # test_image=image.load_img(img_path,target_size=(48,48),color_mode='grayscale')
    # test_image=image.img_to_array(test_image)
    
    test_image = np.expand_dims(test_image, axis=0)
        
    # test_image=test_image.reshape(1,48,48,1)
    classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    images=['angry.jpg','disgust.jpg','fear.png','happy.jpg','neutral.png','sad.jpg','surprise.png']
    result=model.predict(test_image)
    
    y_pred=np.argmax(result[0])
    print('The person facial emotion is:',classes[y_pred])
   
    emotion_prediction = classes[y_pred]
    list =[]
    list.append(images[y_pred])
    list.append(emotion_prediction)
   
    return list    