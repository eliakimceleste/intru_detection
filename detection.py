import numpy as np
import cv2 #, face_recognition
import supervision as sv
from ultralytics import YOLO
import torch
import os

class PersonDetection:
    
    def __init__(self, capture_index, notification):

        self.capture_index = capture_index

        self.notification = notification

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

    # def face_detection(img):
    #     known_image = face_recognition.load_image_file("eli.jpg")
    #     known_image_encoding = face_recognition.face_encodings(known_image)[0]

    #     face_locations = face_recognition.face_locations(img)
    #     #face_encodings = face_recognition.face_encodings(img, face_locations)
    #     unknown_encoding = face_recognition.face_encodings(img, face_locations)
    #     # print(face_locations)
    #     return known_image_encoding, face_locations, unknown_encoding

        #Itérer sur les visages trouvés
        #for (top, right, bottom, left), face_encoding in zip(face_locations, unknown_encoding):
        for (top, left, bottom, right),face_encoding in zip(face_locations, unknown_encoding):
            #cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            #Récupérer seulement le visage à l'instant
            #face_img = img[top:bottom, left:right]
            #Encoder le visage actuelle l'indice 0 pour dire le premier visage vu qu'on a un seul visage
            #unknown_encoding = face_recognition.face_encodings(face_img)#[0]
            #print((top, right,bottom,left))
            
            results = face_recognition.compare_faces([known_image_encoding], face_encoding)
            print(results)
        return results


        

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
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
                
                # Prédiction sur l'image
                results = self.predict(img)
                try:
                    if results:
                        # print(results)
                        # known_image_encoding, face_locations, unknown_encoding = self.face_detection(img)
                        # print(face_locations)
                        # for (top, left, bottom, right),face_encoding in zip(face_locations, unknown_encoding):

                        #     face_detection = face_recognition.compare_faces([known_image_encoding], face_encoding)
                        #     print(face_detection)
                        #     if face_detection[0] == False:
                        #         print("Visage Inconnu")
                        #         face_img = img[top:bottom, left:right]
                        #         cv2.imwrite(f"./images/intruder.jpg",face_img)

                        # img = self.plot_bboxes(results, img)

                        #let's take each person detected and take the frame then save in the images folder
                        for xyxy, track_id in zip(results.xyxy,results.tracker_id):
                            intruImg = img[int(xyxy[1]-25):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                            cv2.imwrite(f"./images/intruder_{track_id}.jpg",intruImg)
                        
                        #Send notification
                        self.notification.send_email(len(results.class_id))
                        print('Envoye')
                        #Vérifiez si plot est une image valide
                        if img is not None and img.any():
                            cv2.imshow('Intruder Detection', img)
                        else:
                          print("No bounding boxes detected.")
                    # known_image_encoding, face_locations, unknown_encoding = self.face_detection(img)
                    # print(face_locations)
                    # for (top, left, bottom, right),face_encoding in zip(face_locations, unknown_encoding):
                    #     face_detection = face_recognition.compare_faces([known_image_encoding], face_encoding)
                    #     print(face_detection)
                    #     if results[0] == True:
                    #         print("C'est bien Eliakim")
                    #         cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                    #     else:
                    #         print("Visage inconnu")

                    
                    cv2.imshow('Intruder Detection', img)
                #cv2.imshow('Intruder Detection', plot)

                    if cv2.waitKey(1) == 27:  # ESC key to break
                        break
                except:
                        continue
        finally:
            cap.release()
            cv2.destroyAllWindows()

                        
    #function to delete file

    def delete_file(path):
        files = os.listdir(path)

        for file in files:
            os.remove(os.path.join(path,file))                        
# Then notification sent, we must delete all previous saved images

# if __name__ == "__main__":
#     obj_detection = ObjectDetection(0)  # 0 pour la webcam intégrée, ou spécifiez l'index de votre caméra
#     obj_detection()

                        