import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mask_model.h5")

face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = frame[y:y+h,x:x+w]
        face = cv2.resize(face,(64,64))
        face = face/255.0
        face = np.reshape(face,(1,64,64,3))

        pred = model.predict(face)

        label = np.argmax(pred)

        if label == 0:
            text = "Mask"
            color = (0,255,0)
        else:
            text = "No Mask"
            color = (0,0,255)

        cv2.putText(frame,text,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

    cv2.imshow("Mask Detection",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()