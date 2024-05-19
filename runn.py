import cv2
import numpy as np
import threading
import time
from tensorflow.keras.models import load_model
from src.utils import Park_classifier


def preprocess_input_frame(input_frame, position, target_size=(180, 180)):
    # Crop the frame to the specified parking spot area
    x, y = position
    crop_frame = input_frame[y:y + 40, x:x + 90]  # Adjust these values if rect_width and rect_height change
    # Resize the cropped frame to match the model's input size
    resized_frame = cv2.resize(crop_frame, target_size)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    resized_frame = np.expand_dims(resized_frame, axis=0)
    resized_frame = resized_frame / 255.0
    return resized_frame


def load_trained_model(model_path):
    # Load the trained model
    model = load_model(model_path)
    return model


def update_status(classifier, model, status_list, frame_lock, frame, interval_seconds):
    while True:
        with frame_lock:
            for idx, pos in enumerate(classifier.car_park_positions):
                # Preprocess the frame for model input
                input_frame = preprocess_input_frame(frame, pos)

                # Make predictions using the model
                predictions = model.predict(input_frame)

                # Determine the predicted class based on the prediction
                status_list[idx] = "Occupied" if predictions[0][0] > 0.5 else "Empty"

        time.sleep(interval_seconds)


def run(video_path, model_path, car_positions_path, interval_seconds=5):
    # Define the parameters
    rect_width, rect_height = 90, 40
    car_park_positions_path = car_positions_path

    # Load the classifier for car park positions
    classifier = Park_classifier(car_park_positions_path, rect_width, rect_height)

    # Load the trained model
    model = load_trained_model(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Set a default fps if unable to read from the video
    frame_duration = int(1000 / fps)

    # Initialize status list
    status_list = ["Unknown"] * len(classifier.car_park_positions)

    # Initialize a lock for accessing the frame
    frame_lock = threading.Lock()

    # Get the first frame for initial processing
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video file.")
        return

    # Start the status update thread
    update_thread = threading.Thread(target=update_status,
                                     args=(classifier, model, status_list, frame_lock, frame, interval_seconds))
    update_thread.daemon = True
    update_thread.start()

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        with frame_lock:
            # Draw rectangles and status text based on the last known status
            for idx, pos in enumerate(classifier.car_park_positions):
                # Define the boundaries
                start = pos
                end = (pos[0] + rect_width, pos[1] + rect_height)

                # Determine the color based on the classification
                color = (0, 0, 255) if status_list[idx] == "Occupied" else (0, 255, 0)

                # Draw the rectangle into the image
                cv2.rectangle(frame, start, end, color, 2)

                # Put the index near the rectangle
                text_position = (start[0], start[1] - 10)  # Adjust as needed for better visibility
                cv2.putText(frame, f'{idx + 1}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame with annotations
        cv2.imshow("Car Park Image (Occupied/Empty)", frame)

        # Control the frame rate
        if cv2.waitKey(frame_duration) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define paths to video and model
    video_path = "data/source/carPark small.mp4"
    model_path = "data/results/trained_model.h5"
    car_positions_path = "data/source/CarParkPos small"

    run(video_path, model_path, car_positions_path, interval_seconds=3)
