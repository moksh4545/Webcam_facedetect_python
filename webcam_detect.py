import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

while True:

    ret, vid = video_capture.read()

    gray_img = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', vid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Video', vid)

video_capture.release()
cv2.destroyAllWindows()
