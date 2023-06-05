import cv2
import dlib

# Load pre-trained models for face detection and facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load pre-trained models for gender and age classification
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")

# List of age and gender labels
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

# Load input image
image = cv2.imread("input_image.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Iterate over detected faces
for face in faces:
    # Determine facial landmarks
    landmarks = predictor(gray, face)

    # Extract face coordinates
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Crop face region of interest
    face_roi = image[y:y+h, x:x+w]

    # Preprocess face for gender classification
    face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Perform gender classification
    gender_net.setInput(face_blob)
    gender_preds = gender_net.forward()
    gender = gender_labels[gender_preds[0].argmax()]

    # Preprocess face for age classification
    face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Perform age classification
    age_net.setInput(face_blob)
    age_preds = age_net.forward()
    age = age_labels[age_preds[0].argmax()]

    # Draw bounding box and labels on the image
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = "{}, {}".format(gender, age)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Display the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
