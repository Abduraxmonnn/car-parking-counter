import cv2
from src.utils import Park_classifier


def run():
    """
    It is a demonstration of the application.
    """

    # defining the params
    rect_width, rect_height = 90, 40
    car_park_positions_path = "data/source/CarParkPos"
    video_path = "data/source/carPark old 1.mp4"

    # creating the classifier instance which uses basic image processes to classify
    classifier = Park_classifier(car_park_positions_path, rect_width, rect_height)

    # Implementation of the classy
    cap = cv2.VideoCapture(video_path)
    while True:
        # reading the video frame by frame
        ret, frame = cap.read()

        # check if there is a retval
        if not ret:
            break

        # processing the frames to prepare classify
        processed_frame = classifier.implement_process(frame)

        # classify the current frame
        classifier.classify(image=frame, prosessed_image=processed_frame)

        # Add counter and draw rectangles with indices
        for idx, pos in enumerate(classifier.car_park_positions):
            # defining the boundaries
            start = pos
            end = (pos[0] + rect_width, pos[1] + rect_height)

            # determine the color based on the classification
            color = (0, 255, 0) if not classifier.is_occupied(idx) else (0, 0, 255)

            # drawing the rectangle into the image
            cv2.rectangle(frame, start, end, color, 2)

            # putting the index near the rectangle
            text_position = (start[0], start[1] - 10)  # Adjust as needed for better visibility
            cv2.putText(frame, f'{idx + 1}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # displaying the results
        cv2.imshow("Car Park Image which drawn", frame)

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
    run()
