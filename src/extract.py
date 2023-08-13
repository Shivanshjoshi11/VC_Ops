import cv2, numpy, os, time
path = r'C:\Users\DELL\OneDrive\Desktop\zo zo zo\image_collection\datasets'  
sub_data = 'sample'

(width, height) = (2000, 1000)
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture('rtsp://admin:TPGEFA@192.168.1.126:554/')
count = 1
while (webcam.isOpened()): 
    ret, image = webcam.read()
    cv2.waitKey(1)
    if image is not None:
        image = cv2.resize(image, (width, height))
        if ret:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            humans = human_cascade.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors=1
            )
            for (x, y, w, h) in humans:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                humans = image[y:y + h, x:x + w]
                cv2.imwrite('% s/% s.png' % (path, count), humans)
                count+=1
    # time.sleep(1)
        
        cv2.imshow('OpenCV', image)
