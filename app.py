import os

import cv2
import numpy as np
import glob
import random
import time

from datetime import datetime

from flask import Flask
from flask_pymongo import PyMongo
from bson.json_util import dumps
from flask import jsonify, request
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename


# memasukan hasil training dan settingan training ke openCV
net = cv2.dnn.readNet("Models_2/yolov3_training_last.weights", "Models_2/yolov3_testing.cfg")

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

app = Flask(__name__)
app.secret_key = "secret key"
app.config["MONGO_URI"] = "mongodb://localhost:27017/bigpro"
mongo = PyMongo(app)


UPLOAD_FOLDER = 'static/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENTIONS = set(['jpg', 'jpeg', 'png'])

@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status': 404,
        'message': 'Not Found: ' + request.url,
    }
    resp = jsonify(message)
    resp.status_code = 404
    return resp

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS

@app.route('/upload', methods=['POST'])
def upload():
    created = datetime.today()

    file = request.files['inputFile']
    filename = secure_filename(file.filename) 

    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        images_path = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        random.shuffle(images_path)
        for img_path in images_path:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (480,360))
            height, width, channels = img.shape

            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            start = time.time()
            outs = net.forward(output_layers)
            end = time.time()
            print("[INFO] Waktu deteksi yolo {:.6f} detik".format(end - start))

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        print(class_id)
                        
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            font = cv2.FONT_HERSHEY_PLAIN
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            unique, counts = np.unique(class_ids, return_counts=True)
            tambah=0
            cv2.rectangle(img, (3, 3), (165, 80), (0,0,255), 1)
            for i in range (len(counts)):
                            cv2.putText(img,str(classes[i])+" = "+str(counts[i]), (5,15+tambah),font,1, (0,0,255), 1)
                            tambah=tambah+15
            print(indexes)
            daftar=[]
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    daftar.append(label)
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                    text = "{}: {:.2f}".format(label, confidences[i])
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1)

            print(daftar)
            mongo.db.log_detection.insert_one({'gambar': filename, 'waktu': created, 'label ': text})
            resp = jsonify(text)
            
            key = cv2.waitKey(0)
            
        cv2.destroyAllWindows()
        return resp
    else:
        resp = jsonify('Gagal upload hanya bisa upload file berformat jpg jpeg dan png')
        return resp

@app.route('/mahasiswa', methods=['GET'])
def get_all_mahasiswa():
	mahasiswa = mongo.db.mahasiswa 
	output = []
	for s in mahasiswa.find():
		output.append({'nim': s['nim'],'name' : s['name'], 'semester' : s['semester'], 'alamat' : s['alamat']})
	return jsonify({'result' : output})

@app.route('/mahasiswa/<nim>', methods=['GET']) 
def get_one_book(nim=None):
  mahasiswa = mongo.db.mahasiswa
  s = mahasiswa.find_one({"nim" : nim})
  if s:
    output = {'nim' : s['nim'], 'nama' : s['nama'], 'semester' : s['semester'], 'alamat' : s['alamat']}
  else:
    output = "No such name"
  return jsonify({'result' : output})


@app.route('/mahasiswa', methods=['POST'])
def add_mahasiswa():
  mahasiswa = mongo.db.mahasiswa
  nim = request.json['nim']
  name = request.json['name']
  semester = request.json['semester']
  alamat = request.json['alamat']
  mahasiswa.insert_one({'nim': nim,'name': name,'semester' : semester, 'alamat': alamat})
  return jsonify({"Pesan": "DATA TERSIMPAN"})

@app.route('/mahasiswa/update', methods=['POST'])
def update_mahasiswa():
    mahasiswa = mongo.db.mahasiswa
    nim = request.json['nim']
    name = request.json['name']
    semester = request.json['semester']
    alamat = request.json['alamat']
    mahasiswa.update_one({'nim':nim},{'$set' : {'name':name,'semester':semester, 'alamat' : alamat}})
    
    
    return jsonify({'result': "Data Sudah Terupdate"})


@app.route('/mahasiswa/delete', methods=['POST'])
def delete_mahasiswa():
  mahasiswa = mongo.db.mahasiswa
  nim = request.json['nim']
  books_id = mahasiswa.delete_one({'nim' : nim})
  return jsonify({'result' : 'success'})

if __name__ == "__main__":
    app.run()