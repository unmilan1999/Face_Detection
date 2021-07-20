import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


def detect(img, face, eye, mouth):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, face, 1.1, 10, color['blue'], "Face")
    if len(coords)==4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        coords = draw_boundary(roi_img, eye, 1.1, 14, color['red'], "Eye")
        coords = draw_boundary(roi_img, mouth, 1.2, 45, color['white'], "Mouth")
    return img



video_capture = cv2.VideoCapture(0)

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

while True:
    _, img = video_capture.read()
    img = detect(img, face, eyes, mouth)
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
