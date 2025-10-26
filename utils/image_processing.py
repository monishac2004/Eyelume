import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_eye_image(image_path):
    cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    eye_cascade = cv2.CascadeClassifier(cascade_path)
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(eyes) > 0
