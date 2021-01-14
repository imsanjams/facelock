import scan 
import os
import numpy as np


subjects = ["", "sanjam singh", ""]


def detect_face(img):
    gray = scan.cvtColor(img, scan.COLOR_BGR2GRAY)
    
    face_cascade= scan.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    if (len(faces)==0):
        return None, None
    
    (x, y, w, h)= faces[0]

    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):

    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for dir_name in dirs:
        
        if not dir_name.startswith("s"):
            continue;
  
        label=int(dir_name.replace("s", ""))

        subject_dir_path=data_folder_path + "/" + dir_name

        subject_images_names=os.listdir(subject_dir_path)
      
        for image_name in subject_images_names:
           
            if image_name.startswith("."):
                continue;
            
            image_path = subject_dir_path + "/" + image_name

            image =scan.imread(image_path)

            scan.imshow("Training on image...", scan.resize(image, (400, 500)))
            scan.waitKey(100)
            

            face, rect=detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    scan.destroyAllWindows()
    scan.waitKey(1)
    scan.destroyAllWindows()
    
    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer=scan.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    scan.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
def draw_text(img, text, x, y):
    scan.putText(img, text, (x, y), scan.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
    img=test_img.copy()
    face, rect=detect_face(img)
    label, confidence=face_recognizer.predict(face)
    label_text=subjects[label]
    
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img


print("Predicting images...")

test_img =scan.imread("pic/ss.jpg")
test_img2=scan.imread("pic/ss1.jpg")

predicted_img1 =predict(test_img1)
predicted_img2 =predict(test_img2)
print("Prediction complete")

scan.imshow(subjects[1], scan.resize(predicted_img1, (400, 500)))
scan.imshow(subjects[2], scan.resize(predicted_img2, (400, 500)))
scan.waitKey(0)
scan.destroyAllWindows()
scan.waitKey(1)
scan.destroyAllWindows()



