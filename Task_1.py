import dlib
import cv2
from matplotlib.pyplot import cla


def hog_detector(vidPath, scaling_factor):
        
    ### HOG face detector
    detector = dlib.get_frontal_face_detector()

    # eye_detector = dlib.get

    # inputImgPath = r'faceImg.jpg'
    cap = cv2.VideoCapture(vidPath)
    while True:
        ret, frame = cap.read()
        # image = cv2.imread(inputImgPath)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        upsample = 1
        faces = detector(rgb, upsample)
        for face_landmark in faces:
            x1 ,y1  = face_landmark.left(), face_landmark.top() 
            x2, y2  = face_landmark.right(), face_landmark.bottom() 
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0),2)

        print("Number of faces ",len(faces))


        cv2.imshow('face',frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


def haar_cascade_face_detection(vidPath, scaling_factor):
    cap = cv2.VideoCapture(vidPath)
    # detector = cv2.CascadeClassifier("data/haarcascade/haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier()
    detector.load(cv2.samples.findFile(cv2.data.haarcascades +"haarcascade_frontalface_default.xml"))
    eyes_detector = cv2.CascadeClassifier()
    eyes_detector.load(cv2.samples.findFile(cv2.data.haarcascades +"haarcascade_eye.xml"))
    while True:
        ret, frame = cap.read()
        if ret:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # rects = classifier.detectMultiScale(gray_img, scaleFactor =scaling_factor, minNeighbors = 5, minSize= (30,30))
            rects = detector.detectMultiScale(gray_img, minNeighbors=7)
            for (x,y,w,h) in rects:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)

            eyes_rects = eyes_detector.detectMultiScale(gray_img, minNeighbors=7)
            for (x,y,w,h) in eyes_rects:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)

            cv2.imshow('Face output', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            print("NO stream")
            break

def main(vidPath):
    print("############# Face detection program ############# ")    
    print("Face detection algorithm :")
    print("1. HOG ")
    print("2. Haar-cascade ")

    algorithm_chosen = input("Please choose algorithm (1 /2) : ")
    while not algorithm_chosen.isnumeric():
        algorithm_chosen = input("Please choose algorithm (1 /2) : ")
    while int(algorithm_chosen)<1 or int(algorithm_chosen)>2:
        print("Choose either 1 or 2 ..")
        algorithm_chosen = input("Please choose algorithm (1 /2) : ")



    scaling_factor = input('Please choose scaling factor for image (default 0.5) : ')
    if len(scaling_factor)==0:
        scaling_factor = 0.5
    else:
        scaling_factor = float(scaling_factor)

    if int(algorithm_chosen)==1:
        hog_detector(vidPath, scaling_factor)
    elif int(algorithm_chosen)==2:
        haar_cascade_face_detection(vidPath, scaling_factor)


if __name__ =="__main__":
    vidPath = r''
    vidPath = 0
    main(vidPath)

