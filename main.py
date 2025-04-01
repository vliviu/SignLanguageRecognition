
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Preprocessing: Convert image to grayscale, and apply thresholding.
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

# 2. Feature Extraction: Use HOG (Histogram of Oriented Gradients) for feature extraction
def extract_features(image):
    # Resize image to a fixed size for consistency
    image_resized = cv2.resize(image, (128, 128))
    
    # HOG feature extraction
    hog = cv2.HOGDescriptor()
    features = hog.compute(image_resized)
    
    return features.flatten()

# 3. Load dataset: Assuming you have a set of labeled images for training
def load_dataset(image_paths, labels):
    features = []
    for img_path in image_paths:
        img = preprocess_image(img_path)
        feature = extract_features(img)
        features.append(feature)
    
    return np.array(features), np.array(labels)

# 4. Train a classifier (e.g., SVM)
def train_classifier(X_train, y_train):
    clf = SVC(kernel='linear')  # Linear SVM
    clf.fit(X_train, y_train)
    return clf

# 5. Predict new gestures
def predict_gesture(clf, image_path):
    img = preprocess_image(image_path)
    features = extract_features(img)
    prediction = clf.predict([features])
    return prediction

# Example workflow
if __name__ == '__main__':
    # Dummy dataset paths and labels
    image_paths = ['gesture1.png', 'gesture2.png', 'gesture3.png']  # Replace with your dataset paths
    labels = [0, 1, 2]  # Replace with corresponding labels for gestures

    # Load dataset
    X, y = load_dataset(image_paths, labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = train_classifier(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))

    # Predict a new gesture
    test_image = 'new_gesture.png'  # Replace with a new gesture image
    predicted_label = predict_gesture(clf, test_image)
    print("Predicted Label:", predicted_label)
