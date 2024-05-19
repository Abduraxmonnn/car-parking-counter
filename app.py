import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.utils import Park_classifier


def preprocess_input_frame(input_frame, position, target_size=(180, 180)):
    """
    Preprocesses a cropped area of the input frame for model prediction.

    Args:
        input_frame (np.ndarray): The input frame from the video.
        position (tuple): The top-left (x, y) coordinates of the parking spot to crop.
        target_size (tuple): The target size to resize the cropped area to. Default is (180, 180).

    Returns:
        np.ndarray: The preprocessed frame ready for model input.
    """
    x, y = position
    crop_frame = input_frame[y:y + 40, x:x + 90]  # Adjust these values if rect_width and rect_height change
    resized_frame = cv2.resize(crop_frame, target_size)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    resized_frame = np.expand_dims(resized_frame, axis=0)
    resized_frame = resized_frame / 255.0
    return resized_frame


def load_trained_model(model_path):
    """
    Loads the trained Keras model from the given path.

    Args:
        model_path (str): The path to the trained model file.

    Returns:
        tensorflow.keras.Model: The loaded model.
    """
    model = load_model(model_path)
    return model


def initial_status_update(classifier, model, frame):
    """
    Initializes the status of all parking spots by making predictions on the initial frame.

    Args:
        classifier (Park_classifier): The classifier instance with car park positions.
        model (tensorflow.keras.Model): The trained model for prediction.
        frame (np.ndarray): The initial frame from the video.

    Returns:
        list: A list of statuses ("Occupied" or "Empty") for each parking spot.
    """
    status_list = ["Unknown"] * len(classifier.car_park_positions)
    for idx, pos in enumerate(classifier.car_park_positions):
        input_frame = preprocess_input_frame(frame, pos)
        predictions = model.predict(input_frame)
        status_list[idx] = "Occupied" if predictions[0][0] > 0.5 else "Empty"
    return status_list


def update_status(classifier, model, status_list, frame, start_idx, end_idx):
    """
    Updates the status of a subset of parking spots based on the model's predictions.

    Args:
        classifier (Park_classifier): The classifier instance with car park positions.
        model (tensorflow.keras.Model): The trained model for prediction.
        status_list (list): The current list of statuses for each parking spot.
        frame (np.ndarray): The current frame from the video.
        start_idx (int): The starting index of the parking spots to update.
        end_idx (int): The ending index of the parking spots to update.
    """
    positions = classifier.car_park_positions[start_idx:end_idx]

    for idx, pos in enumerate(positions, start=start_idx):
        input_frame = preprocess_input_frame(frame, pos)
        predictions = model.predict(input_frame)
        status_list[idx] = "Occupied" if predictions[0][0] > 0.5 else "Empty"


def run(video_path, model_path, car_positions_path, interval_seconds=3):
    """
    Processes and displays a video with parking spot status updates at regular intervals.

    Before the play video code crops every frame and set status after that start playing video
    and after playing video crop frames by 1/10 every some interval.

    This function performs the following steps:
    1. Loads the trained model.
    2. Initializes the status of all parking spots by processing the first frame.
    3. Plays the video, periodically updating the status of parking spots in chunks.

    Args:
        video_path (str): The path to the video file.
        model_path (str): The path to the trained model file.
        interval_seconds (int): The interval in seconds between status updates for chunks of parking spots.
    """
    # Define the parameters
    rect_width, rect_height = 90, 40
    car_park_positions_path = "data/source/CarParkPos"

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

    # Get the first frame for initial processing
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video file.")
        return

    # Initial status update for all parking spots
    status_list = initial_status_update(classifier, model, frame)

    # Reset the video capture to start playing the video from the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize time and index range
    start_time = time.time()
    num_positions = len(classifier.car_park_positions)
    update_chunk = max(1, num_positions // 10)  # At least one position per chunk
    current_idx = 0

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= interval_seconds:
            # Update status for the current chunk
            end_idx = min(current_idx + update_chunk, num_positions)
            update_status(classifier, model, status_list, frame, current_idx, end_idx)

            # Move to the next chunk
            current_idx = end_idx % num_positions
            start_time = current_time

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

    # Run the main function
    run(video_path, model_path, car_positions_path)
