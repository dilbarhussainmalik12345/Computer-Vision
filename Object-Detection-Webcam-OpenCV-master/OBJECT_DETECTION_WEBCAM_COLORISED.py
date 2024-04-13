#Colorised
import cv2

# getting vid from cv2.VideoCapture 
video = cv2.VideoCapture(0)
a = 1
path = 'C:\\Users\\kuk\\OBJECT_DETECTION\\OBJECT_DETECTION_WEBCAM\\haar_cascades'

scale_factor = 1
while True:
    a = a +1    
    face_cascade = cv2.CascadeClassifier(path + "\\haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(path +"\\haarcascade_eye.xml")
    upperbody_cascade = cv2.CascadeClassifier(path + "\\haarcascade_upperbody.xml")
    smile_cascade = cv2.CascadeClassifier(path+"\\haarcascade_smile.xml")
    profileface_cascade = cv2.CascadeClassifier(path+"\\haarcascade_profileface.xml")
    lowerbody_cascade = cv2.CascadeClassifier(path+"\\haarcascade_lowerbody.xml")
    fullbody_cascade = cv2.CascadeClassifier(path+"\\haarcascade_fullbody.xml")
    frontalface_alt_cascade = cv2.CascadeClassifier(path+"\\haarcascade_frontalface_alt.xml")
    # check is a boolian operator that returns TRUE if webcam is working
    # frame gets the frame(imgs) from the vid camera
    check, frame = video.read() 
#    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)    
    img = cv2.resize(frame, (int(frame.shape[1]/scale_factor),int(frame.shape[0]/scale_factor)))    
    
    faces = face_cascade.detectMultiScale(img, 1.05, 5)
    eyes = eye_cascade.detectMultiScale(img, 1.05, 5)
    upperbody = upperbody_cascade.detectMultiScale(img, 1.05, 5)
    smile = smile_cascade.detectMultiScale(img, 1.05, 5)
    profileface = profileface_cascade.detectMultiScale(img, 1.05, 5)
    lowerbody = lowerbody_cascade.detectMultiScale(img, 1.05, 5)
    fullbody = fullbody_cascade.detectMultiScale(img, 1.05, 5)
    frontalface_alt = frontalface_alt_cascade.detectMultiScale(img, 1.05, 5)
    
    for x,y,w,h in faces:
        rec_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0) ,0)
        face_text = cv2.putText(img, "FACE",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(0,255,0))
        
    for x,y,w,h in eyes:
        eye_img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)
        eyes_text = cv2.putText(img, "eyes",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(225,0,0))
        
    for x,y,w,h in upperbody:
        upperbody_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 3)
        upperbody_text = cv2.putText(img, "upper body",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(0,0,255))
        
    for x,y,w,h in smile:
        smile_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,100,0), 3)
        smile_text = cv2.putText(img, "smile",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(0,100,0))
        
    for x,y,w,h in profileface:
        profile_img = cv2.rectangle(img, (x,y), (x+w, y+h), (100,0,0), 3)
        profile_text = cv2.putText(img, "profile_",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(100,0,0))
        
    for x,y,w,h in lowerbody:
        lowerbody_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,100), 3)
        lowerbody_text = cv2.putText(img, "lowerbody",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(0,0,100))
        
    for x,y,w,h in fullbody:
        fullbody_img = cv2.rectangle(img, (x,y), (x+w, y+h), (150,0,0), 3)
        fullbody_text = cv2.putText(img, "fullbody",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(150,0,0))
        
    for x,y,w,h in frontalface_alt:
        frontalface_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,150,0), 3)
        frontalface_text = cv2.putText(img, "frontal_face",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(100,150,0))
    
    cv2.imshow('capturing', img)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
print(a)
video.release

cv2.destroyAllWindows()