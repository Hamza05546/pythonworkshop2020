import cv2

video = cv2.VideoCapture(r"/home/aiktc/Desktop/hamza/imagecoding/faceDetection.mp4")
#print(type(video))
#print(type(check))
check=True
while check:
    check, frame =video.read()
    # cv2.imshow("video ka frame",frame)  #for normal video
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    img = frame
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #BGR TO GRAY TO CHANGE FROM COLOR TO BLACK AND WHITE

    faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.6,minNeighbors=5)  #scale should be btwn 

    for x,y,w,h in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)


    resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

    cv2.imshow("gray",resized)     #for face detect video.
    key = cv2.waitKey(10)
    if(key == ord('q')):
        break



cv2.destroyAllWindows()
video.release()
