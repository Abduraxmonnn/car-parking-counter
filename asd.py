import cv2
import numpy as np
from keras.models import load_model
from src.utils import Park_classifier


def preprocess_input_frame(input_frame, target_size=(180, 180)):
    # Resize the frame and preprocess for model input
    resized_frame = cv2.resize(input_frame, target_size)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    resized_frame = np.expand_dims(resized_frame, axis=0)
    resized_frame = resized_frame / 255.0
    return resized_frame


def load_trained_model(model_path):
    # Load the trained model
    model = load_model(model_path)
    return model


def run(video_path, model_path):
    """
    It is a demonstration of the application.
    """

    # defining the params
    rect_width, rect_height = 90, 40
    car_park_positions_path = "data/source/CarParkPos"

    # creating the classifier instance which uses basic image processes to classify
    classifier = Park_classifier(car_park_positions_path, rect_width, rect_height)

    model = load_trained_model(model_path)

    # Implementation of the classy
    cap = cv2.VideoCapture(video_path)
    while True:
        # reading the video frame by frame
        ret, frame = cap.read()

        # check if there is a retval
        if not ret:
            break

        # Preprocess the frame for model input
        input_frame = preprocess_input_frame(frame)

        # Make predictions using the model
        predictions = model.predict(input_frame)

        # Determine the predicted class based on the prediction
        status = "Occupied" if predictions[0][0] > 0.5 else "Empty"

        # # processing the frames to prepare classify
        # processed_frame = classifier.implement_process(frame)
        #
        # # classify the current frame
        # classifier.classify(image=frame, prosessed_image=processed_frame)

        # Add counter and draw rectangles with indices
        for idx, pos in enumerate(classifier.car_park_positions):
            # defining the boundaries
            start = pos
            end = (pos[0] + rect_width, pos[1] + rect_height)

            # determine the color based on the classification
            # color = (0, 255, 0) if not classifier.is_occupied(idx) else (0, 0, 255)
            color = (0, 255, 0) if not status == "Occupied" else (0, 0, 255)

            # drawing the rectangle into the image
            cv2.rectangle(frame, start, end, color, 2)

            # putting the index near the rectangle
            text_position = (start[0], start[1] - 10)  # Adjust as needed for better visibility
            cv2.putText(frame, f'{idx + 1}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # displaying the results
        cv2.imshow("Car Park Image which drawn According to empty or occupied", frame)

        # exit condition
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break

        if k & 0xFF == ord('s'):
            cv2.imwrite("data/images/output.jpg", frame)

    # re-allocating sources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define paths to video and model
    video_path = "data/source/carPark old 1.mp4"
    model_path = "data/results/trained_model.h5"

    # Run the main function
    run(video_path, model_path)
