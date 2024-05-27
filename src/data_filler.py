import cv2
import os
import time
from src.utils import Park_classifier


def data_filler():
    """
    It is a demonstration of the application.
    """

    # defining the params
    rect_width, rect_height = 90, 40
    carp_park_positions_path = "../data/source/CarParkPos small"
    video_path = "../data/source/carPark small.mp4"
    output_dir = "../data/dataset/new_images/"

    # creating the classifier instance which uses basic image processes to classify
    classifier = Park_classifier(carp_park_positions_path, rect_width, rect_height)

    # Create new_images directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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

                # Generate a unique filename based on the current timestamp
                timestamp = int(time.time())  # Current Unix timestamp
                output_filename = f"frame_{timestamp}_pos_{idx:05d}.jpg"
                output_path = os.path.join(output_dir, output_filename)

                # Save the ROI to the new_images directory
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
