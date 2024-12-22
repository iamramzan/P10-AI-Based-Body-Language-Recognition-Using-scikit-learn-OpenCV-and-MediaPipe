# P10: AI-Based Body Language Recognition Using scikit-learn, OpenCV, and MediaPipe

import mediapipe as mp  # Importing MediaPipe library for pose, hand, and face detection
import cv2  # Importing OpenCV library for video capture and image processing

# Define helpers for drawing landmarks and accessing the holistic model
mp_drawing = mp.solutions.drawing_utils  # Utility functions for drawing landmarks
mp_holistic = mp.solutions.holistic  # Access holistic solutions for pose, hand, and face

# Initialize webcam feed
cap = cv2.VideoCapture(1)  # Capture video from the webcam (index 1 for external webcam)

# Initiate the holistic model with minimum detection and tracking confidence
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():  # Continue processing while the webcam feed is open
        ret, frame = cap.read()  # Read a frame from the webcam

        # Convert the frame from BGR to RGB for holistic model processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Improve performance by making the image read-only

        # Process the image with the holistic model to detect landmarks
        results = holistic.process(image)

        # Convert the image back to BGR for rendering the output
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_holistic.FACEMESH_CONTOURS,  # Draw contours for the face
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),  # Style for landmarks
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)   # Style for connections
        )

        # Draw right hand landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.right_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,  # Draw connections between hand landmarks
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )

        # Draw left hand landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.left_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,  # Draw connections between pose landmarks
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Show the processed video feed with landmarks
        cv2.imshow('Raw Webcam Feed', image)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release webcam resources and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Extract the visibility of the first face landmark (optional line)
results.face_landmarks.landmark[0].visibility

# Coordinate Extraction and CSV Initialization
import csv  # Library for working with CSV files
import os  # Library for interacting with the operating system
import numpy as np  # Library for numerical computations

# Calculate the total number of landmarks (pose + face)
num_coords = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)

# Define the column headers for the CSV file
landmarks = ['class']  # First column will hold the class name
for val in range(1, num_coords + 1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]  # Add x, y, z, and visibility columns for each landmark

# Create the CSV file and write the headers
with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)  # Write headers to the CSV file

# Data Collection
class_name = "Awake"  # Define the class name for this data collection session

cap = cv2.VideoCapture(1)  # Initialize webcam
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()  # Capture a frame from the webcam

        # Convert the frame to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Improve performance by making the image read-only

        # Process the frame with the holistic model
        results = holistic.process(image)

        # Convert the frame back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks on the image (same as earlier)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Extract and save coordinates to the CSV file
        try:
            # Extract pose landmarks and flatten into a row
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract face landmarks and flatten into a row
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # Combine pose and face rows
            row = pose_row + face_row

            # Append the class name to the row
            row.insert(0, class_name)

            # Write the row to the CSV file
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)  # Save the row to the CSV file

        except:
            pass  # Ignore errors if landmarks are not detected

        # Display the video feed with landmarks
        cv2.imshow('Raw Webcam Feed', image)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release webcam resources and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Data Preparation and Model Training
import pandas as pd  # Library for data manipulation
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets

# Load the dataset from the CSV file
df = pd.read_csv('coords.csv')

df.head()

df.tail()

# Split data into features (X) and target labels (y)
X = df.drop('class', axis=1)  # Drop the 'class' column to get the feature set
y = df['class']  # The target labels (class)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)  # 70% training, 30% testing

y_test

# Model Training with Pipelines
from sklearn.pipeline import make_pipeline  # Create machine learning pipelines
from sklearn.preprocessing import StandardScaler  # Normalize features

# Import classifiers
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Define pipelines for each model
pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression()),  # Logistic Regression
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),      # Ridge Classifier
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),  # Random Forest
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),  # Gradient Boosting
}

# Train each model and store in a dictionary
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)  # Train the model
    fit_models[algo] = model  # Save the trained model

fit_models


fit_models['rc'].predict(X_test)

# Model Evaluation and Saving
from sklearn.metrics import accuracy_score  # Metric for evaluating model performance
import pickle  # Library for saving and loading models

# Evaluate each model's accuracy
for algo, model in fit_models.items():
    yhat = model.predict(X_test)  # Make predictions on the test set
    print(algo, accuracy_score(y_test, yhat))  # Print model accuracy

# Save the best model (Random Forest in this case) to a pickle file
with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)

# Load the model for later use
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)
    
model

# Live Prediction Using the Trained Model
cap = cv2.VideoCapture(1)  # Initialize webcam

# Load the trained model from the pickle file
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the holistic model for capturing landmarks
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()  # Capture a frame from the webcam

        # Convert the frame to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Improve performance by making the image read-only

        # Process the frame with the holistic model to detect landmarks
        results = holistic.process(image)

        # Convert the frame back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks on the frame (same as in the data collection step)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Perform prediction only if landmarks are detected
        try:
            # Extract pose landmarks and flatten them into a row
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract face landmarks and flatten them into a row
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # Combine pose and face rows into a single feature row
            row = pose_row + face_row

            # Convert the row to a NumPy array and reshape for the model
            X = pd.DataFrame([row])  # Wrap the row into a DataFrame

            # Make a prediction using the trained model
            body_language_class = model.predict(X)[0]  # Predict the class
            body_language_prob = model.predict_proba(X)[0]  # Predict probabilities for each class

            # Display the predicted class and its probability on the video feed
            cv2.putText(image, 'CLASS: {}'.format(body_language_class), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, 'PROB: {}'.format(np.max(body_language_prob)), (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except:
            pass  # Ignore errors if landmarks are not detected

        # Display the live video feed with landmarks and predictions
        cv2.imshow('Raw Webcam Feed', image)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release webcam resources and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()



















