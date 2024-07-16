import numpy as np
import cv2 #, face_recognition
import supervision as sv
from ultralytics import YOLO
import torch
import os
import threading
from queue import Queue



class PersonDetection:

    def __init__(self):
        pass
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
           )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', device)
        self.model = self.load_model()

    
    def load_model(self):
        model = YOLO('yolov8m.pt')
        model.fuse()
        return model

    def predict(self,img):

        results = self.model.track(img, persist=True,tracker="bytetrack.yaml")[0]

        # Convert result to Supervision Detection object
        detections = sv.Detections.from_ultralytics(results)
        print(len(detections))

        # In Yolov8 model, objects with class_id 0 refer to a person. So, we should filter objects detected to only consider person
        return detections[detections.class_id == 0]
    
    def plot_bboxes(self,detections, img):
        labels = [f"Intruder #{track_id}" for track_id in detections.tracker_id if len(detections.tracker_id) > 0]
        labels = [f"Intruder#{track_id} {self.model.model.names[class_id]} {conf:0.2f}"
                for track_id, class_id, conf in zip(detections.tracker_id, detections.class_id, detections.confidence)
        ]
        
        # Add the box and tehe labels to the image
        annotated_image = self.box_annotator.annotate(scene=img, detections=detections,labels=labels)
        return annotated_image
    
    def capture_frames(self, camera_url, frame_queue):
        cap = cv2.VideoCapture(camera_url)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        frame_count = 0
        print(cap)
        try:
            while True:
                ret, img = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                if img is None or img.size == 0:
                    print("Empty frame or invalid image")
                    continue
                frame_queue.put(img)  # Ajoute le frame à la queue
        finally:
            cap.release()
            cv2.destroyAllWindows()


    def detection(self, frame_queue):
        while True:
            img = frame_queue.get()  # Récupère le frame depuis la queue
            # Prédiction sur l'image
            results = self.predict(img)
            try:
                if results:
                    img = self.plot_bboxes(results, img)

                    #let's take each person detected and take the frame then save in the images folder
                    for xyxy, track_id in zip(results.xyxy,results.tracker_id):
                        intruImg = img[int(xyxy[1]-25):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                        cv2.imwrite(f"./images/intruder_{track_id}.jpg",intruImg)

                #Afficher la frame
                cv2.imshow('Intruder Detection', img)
                if cv2.waitKey(1) == 27:  # ESC key to break
                    break
            except:
                    continue
    

camera_urls = ['0']
#detection = PersonDetection()
# Créez des queues pour chaque caméra
frame_queues = [Queue() for _ in camera_urls]

# Créez et lancez un thread pour la capture de chaque caméra
capture_threads = []
for i, url in enumerate(camera_urls):
    thread = threading.Thread(target=PersonDetection().capture_frames, args=(url, frame_queues[i]))
    thread.start()
    capture_threads.append(thread)

# Créez et lancez un thread pour la détection d'objets
detect_threads = []
for frame_queue in frame_queues:
    thread = threading.Thread(target=PersonDetection().detection, args=(frame_queue,))
    thread.start()
    detect_threads.append(thread)

# Attendez que tous les threads de capture se terminent
for thread in capture_threads:
    thread.join()

# Attendez que tous les threads de détection se terminent
for thread in detect_threads:
    thread.join()