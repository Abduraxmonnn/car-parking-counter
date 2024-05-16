import cv2
from src.utils import Coordinate_denoter


def demostration(
        horizontal: bool = False,
        rectangle_width: int = 40,
        rectangle_height: int = 10
):
    """
    It is the demonstration of the car_park_coordinate_generator.
    """

    # creating the Coordinate_generator instance for extracting the car park coordinates
    coordinate_generator = Coordinate_denoter()

    # reading and initialing the coordinates
    coordinate_generator.read_positions()

    # setting the initial variables
    image_path = "data/source/example_image.png"
    rect_width, rect_height = (rectangle_width, rectangle_height) if horizontal else (rectangle_height, rectangle_width)

    # serving the GUI window until user terminates it
    while True:
        # refreshing the image
        image = cv2.imread(image_path)

        # drawing the current car park coordinates with index
        for idx, pos in enumerate(coordinate_generator.car_park_positions):
            # defining the boundaries
            start = pos
            end = (pos[0] + rect_width, pos[1] + rect_height)

            # drawing the rectangle into the image
            cv2.rectangle(image, start, end, (0, 0, 255), 2)

            # putting the index near the rectangle
            text_position = (start[0], start[1] - 10)  # Adjust as needed for better visibility
            cv2.putText(image, f'{idx + 1}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Image", image)

        # linking the mouse callback
        cv2.setMouseCallback("Image", coordinate_generator.mouseClick)

        # exit condition
        if cv2.waitKey(1) == ord("q"):
            break

    # re-allocating the sources
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demostration(horizontal=True, rectangle_width=80, rectangle_height=40)
