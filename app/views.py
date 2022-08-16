import os
from flask import render_template, request, Response
from flask import redirect, url_for
from PIL import Image
from app.utils import pipeline_model
from app.camera import Video

UPLOAD_FLODER = 'static/uploads'
def base():
    return render_template('base.html')

def index():
    return render_template('index.html')

def faceapp():
    return render_template('faceapp.html')

def getwidth(path):
    
    img = Image.open(path)
    size = img.size # width and height
    aspect = size[0]/size[1] # width / height
    w = 300 * aspect
    return int(w)

def emotion():

    if request.method == "POST":
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(UPLOAD_FLODER,filename)
        f.save(path)
        w = getwidth(path)
        
        # prediction (pass to pipeline model)
        result = pipeline_model(path,filename)

        return render_template('emotion.html',fileupload=True,img_name=filename, w=w, result_img=result[0],result_text=result[1])

    return render_template('emotion.html',fileupload=False,img_name="freeai.png")

def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')

def video():
    
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

def camera():
    return render_template('capture.html')
   



  