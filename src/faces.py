import cv2, numpy, os, imutils, time
path = r'C:\Users\DELL\OneDrive\Desktop\zo zo zo\image_collection\datasets'  
sub_data = 'sample'

# (width, height) = (500, 1000)
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
webcam = cv2.VideoCapture(0)
count = 1
while (webcam.isOpened()): 
    ret, image = webcam.read()
    image = imutils.resize(image, width=min(900, image.shape[1]))

    key = cv2.waitKey(1)
    # if 0xFF == ord('e'):
    #     break
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(
        image,
        1.9,
        1
    )
    for (x, y, w, h) in humans:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        humans = gray[y:y + h, x:x + w]
        cv2.imwrite('% s/% s.png' % (path, count), humans)
        count+=1
    time.sleep(1)
        
    cv2.imshow('OpenCV', image)