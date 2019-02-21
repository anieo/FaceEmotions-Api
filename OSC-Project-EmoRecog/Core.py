from keras.models import model_from_json
from keras.optimizers import SGD
import numpy as np
import cv2
import os
from time import sleep


model = model_from_json(open('./model_weights/Face_model_architecture.json').read())
# model.load_weights('_model_weights.h5')
model.load_weights('./model_weights/Face_model_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    # print x , y, w ,h
    horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
    vertical_offset = np.int(np.floor(offset_coefficients[1] * h))

    extracted_face = gray[y + vertical_offset:y + h,
                     x + horizontal_offset:x - horizontal_offset + w]
    # print extracted_face.shape
    new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0],
                                               48. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(np.float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face


from scipy.ndimage import zoom


def detect_face(frame):
    cascPath = "./model_weights/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(48, 48),
        # flags=cv2.cv.CV_HAAR_FEATURE_MAX
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return gray, detected_faces


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

Estimated=0.0;
Div=0.0
Sample=0.0
Typical_value=0.125
prediction_result=0.0
avg_values = []

dir=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(dir,"Input")

for root,dirs,files in os.walk(image_dir):
    for file in files :

        if file.endswith("png")or file.endswith("jpg")or file.endswith("jpeg"):

            path = os.path.join(root, file)
frame = cv2.imread(path,1)
gray, detected_faces = detect_face(frame)
gray, detected_faces = detect_face(frame)

face_index = 0

# predict output
for face in detected_faces:
    (x, y, w, h) = face
    if w > 100:
        # Mark Rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        extracted_face = extract_face_features(gray, face, (0.075, 0.05))  # (0.075, 0.05)


        prediction_result = model.predict_classes(extracted_face.reshape(1, 48, 48, 1))
        Sample = prediction_result
        Estimated = (1 - Typical_value) * Estimated + Typical_value * Sample
        div = (1 - Typical_value) * Div + Typical_value * abs(Sample - Estimated)
        #prediction_result = Estimated + 4 * Div
        avg_values.append(prediction_result)
        if len(avg_values) > 10:
            avg_values.pop(0)
        av = sum(avg_values)
        #prediction_result = av / len(avg_values)
        #prediction_result = int(round(prediction_result, 5))


        if prediction_result == 3:
            cv2.putText(frame, "Happy", (x, y), cv2.FONT_ITALIC, 2, 155, 10)
        elif prediction_result == 0:
            cv2.putText(frame, "Angry", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
        elif prediction_result == 1:
            cv2.putText(frame, "Disgust", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
        elif prediction_result == 2:
            cv2.putText(frame, "Fear", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
        elif prediction_result == 4:
            cv2.putText(frame, "Sad", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
        elif prediction_result == 5:
            cv2.putText(frame, "Surprise", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
        elif prediction_result == 6:
            cv2.putText(frame, "Neutral", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
        cv2.imwrite('../static/'+file, frame)
        file = open("../static/result.txt", "w")
        file.write(str(prediction_result[0]))
        file.close()

