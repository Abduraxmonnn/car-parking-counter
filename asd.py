import cv2
import numpy as np
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


def run(video_path, model_path, car_positions_path):
    # Define the parameters
    rect_width, rect_height = 90, 40
    car_park_positions_path = car_positions_path

    # Load the classifier for car park positions
    classifier = Park_classifier(car_park_positions_path, rect_width, rect_height)

    # Load the trained model
    model = load_trained_model(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Add counter and draw rectangles with indices based on the predicted status
        for idx, pos in enumerate(classifier.car_park_positions):
            # Preprocess the frame for model input
            input_frame = preprocess_input_frame(frame, pos)

            # Make predictions using the model
            predictions = model.predict(input_frame)

            # Determine the predicted class based on the prediction
            if predictions[0][0] > 0.5:
                status = "Occupied"
                color = (0, 0, 255)  # Red color for "Occupied"
            else:
                status = "Empty"
                color = (0, 255, 0)  # Green color for "Empty"

            # Define the boundaries
            start = pos
            end = (pos[0] + rect_width, pos[1] + rect_height)

            # Draw the rectangle into the image
            cv2.rectangle(frame, start, end, color, 2)

            # Put the index near the rectangle
            text_position = (start[0], start[1] - 10)  # Adjust as needed for better visibility
            cv2.putText(frame, f'{idx + 1}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame with annotations
        cv2.imshow("Car Park Image (Occupied/Empty)", frame)

        # Exit condition
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define paths to video and model
    # video_path = "data/source/carPark small.mp4"
    # model_path = "data/results/trained_model.h5"
    # car_positions_path = "data/source/CarParkPos small"

    video_path = "data/source/carPark small.mp4"
    model_path = "data/model/trained_model.h5"
    car_positions_path = "data/source/CarParkPos small"

    # Run the main function
    run(video_path, model_path, car_positions_path)

# rectangle qancha kop bosa video shuncha kamayib ketvoti sababi har doim preprocess_input_frame() function ishlavokani
# Shuni nima qsa boladi ?
