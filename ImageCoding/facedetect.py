import cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("pool.jpg")
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #BGR TO GRAY TO CHANGE FROM COLOR TO BLACK AND WHITE

faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.01,minNeighbors=5)  #scale should be btwn 

for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)


print(type(faces))
print(faces)

resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

cv2.imshow("gray",resized)
cv2.waitKey(100)
cv2.destroyAllWindows
