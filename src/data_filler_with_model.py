import cv2
import os
import time
import numpy as np
from tensorflow.keras.models import load_model
from src.utils import Park_classifier


def preprocess_input_frame(input_frame, target_size=(180, 180)):
    # Resize the frame and preprocess for model input
    resized_frame = cv2.resize(input_frame, target_size)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    resized_frame = np.expand_dims(resized_frame, axis=0)
    resized_frame = resized_frame / 255.0
    return resized_frame


def data_filler():
    """
    It is a demonstration of the application.
    """

    # defining the params
    rect_width, rect_height = 90, 40
    carp_park_positions_path = "../data/source/CarParkPos small"
    video_path = "../data/source/carPark small.mp4"
    empty_dir = "../data/dataset/new_images/empty"
    occupied_dir = "../data/dataset/new_images/occupied"
    model_path = "../data/results/trained_model.h5"

    # Load the trained model
    model = load_model(model_path)

    # creating the classifier instance which uses basic image processes to classify
    classifier = Park_classifier(carp_park_positions_path, rect_width, rect_height)

    # Create directories if they don't exist
    os.makedirs(empty_dir, exist_ok=True)
    os.makedirs(occupied_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize timing variables
    start_time = time.time()
    capture_interval = 1  # Interval for frame extraction in seconds

    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if there is a valid frame
        if not ret:
            break

        # Process the frame to prepare for classification
        processed_frame = classifier.implement_process(frame)

        # Classify the current frame
        classifier.classify(frame, processed_frame)

        # Check if it's time to capture a new frame
        elapsed_time = time.time() - start_time
        if elapsed_time >= capture_interval:
            # Reset timer
            start_time = time.time()

            # Extract and save ROIs based on classification
            for idx, pos in enumerate(classifier.car_park_positions):
                x, y = pos[0], pos[1]

                # Define the region of interest (ROI) based on the rectangle position
                roi = frame[y:y + rect_height, x:x + rect_width]

                # Preprocess the ROI for prediction
                preprocessed_roi = preprocess_input_frame(roi)

                # Make prediction using the model
                prediction = model.predict(preprocessed_roi)

                # Determine the predicted class based on the prediction
                status = "occupied" if prediction[0][0] > 0.5 else "empty"

                # Generate a unique filename based on the current timestamp
                timestamp = int(time.time())  # Current Unix timestamp
                output_filename = f"frame_{timestamp}_pos_{idx:05d}.jpg"

                if status == "occupied":
                    output_path = os.path.join(occupied_dir, output_filename)
                else:
                    output_path = os.path.join(empty_dir, output_filename)

                # Save the ROI to the respective directory
                cv2.imwrite(output_path, roi)

                # Increment frame count
                frame_count += 1

        # Display the frame (optional, for visualization)
        cv2.imshow("Car Park Image (Occupied/Empty)", frame)

        # Check for key press events
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):  # Quit if 'q' is pressed
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    data_filler()
