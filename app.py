from flask import Flask, request, send_file, jsonify
from PIL import Image
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from io import BytesIO
import joblib

app = Flask(__name__)

def detect_face(img):
    # Load Haar Cascades
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert image to grayscale
    if len(img.shape) == 3:  # If image is BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # If image is already grayscale
        gray = img

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Check if no faces detected
    if len(faces) == 0:
        return None

    # Crop detected faces and save in list
    cropped_faces = []
    for (x, y, w, h) in faces:
        face_crop = gray[y:y + h, x:x + w]
        cropped_faces.append(face_crop)

    return cropped_faces

def extract_glcm_features(image):
    # Quantize image to reduce intensity levels
    levels = 8
    image_quantized = (image // (256 // levels)).astype('uint8')

    # Define distances and angles
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°

    # Calculate GLCM
    glcm = graycomatrix(image_quantized, 
                       distances=distances,
                       angles=angles,
                       levels=levels,
                       symmetric=True,
                       normed=True)

    # Extract GLCM properties
    properties = ['dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'mean', 'variance', 'std', 'entropy']
    features = []

    # Calculate average properties for all angles
    for prop in properties:
        feature = graycoprops(glcm, prop)
        features.extend(feature.flatten())

    return np.array(features)

def process_image(img):
    features_list = []
    
    # Detect faces
    faces = detect_face(img)
    
    if faces is None:
        return False
    else:
        for face in faces:
            # Resize face to 128x128 for consistency
            resized_face = cv2.resize(face, (128, 128))
            
            # Extract GLCM features
            glcm_features = extract_glcm_features(resized_face)
            features_list.append(glcm_features)
    
    columns = [f'feature_{i}' for i in range(len(features_list[0]))]
    return pd.DataFrame(features_list, columns=columns)


def predict(dataframe):
    df = dataframe

    model, refs_cols, target = joblib.load("GLCM_SVM_Model.pkl")

    X_new = df[refs_cols]

    prediction = model.predict(X_new)
    
    # Melakukan prediksi
    print("Prediction:", prediction)
    
    # Optionally, return the prediction
    return prediction

@app.route('/process-image', methods=['POST'])
def classify():
    try:
        # Get image from request
        img_file = request.files['image']
        
        # Convert to OpenCV format
        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)

        dataframe = process_image(img)
        
        # Detect faces
        # faces = detect_face(img)
        
        if dataframe is False:
            return jsonify({'error': 'No face detected in the image'}), 400
        
        else:
            prediction = predict(dataframe)
            return jsonify({
                'status': 'success',
                'prediction': prediction.tolist()[0],
            })
        
        # features_list = []
        
        # # Process each detected face
        # for face in faces:
        #     # Resize face to 128x128 for consistency
        #     resized_face = cv2.resize(face, (128, 128))
            
        #     # Extract GLCM features
        #     glcm_features = extract_glcm_features(resized_face)
        #     features_list.append(glcm_features)
        
        # # Create DataFrame with features
        # columns = [f'feature_{i}' for i in range(len(features_list[0]))]
        # df = pd.DataFrame(features_list, columns=columns)
        
        # # Here you would add your model prediction
        # # prediction = model.predict(df)
        
        # # For now, just return the features as JSON
        # return jsonify({
        #     'status': 'success',
        #     'features': df.to_dict(orient='records'),
        #     'num_faces_detected': len(faces)
        # })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)