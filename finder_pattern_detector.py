import cv2
import numpy as np


class FinderPatternDetector:
    """
    This class represents a finder pattern detector used for detecting patterns in images.

    Attributes:
        centers (list): A list to store the detected centers of finder patterns.
        module_size (list): A list to store the module sizes of detected finder patterns.
        path_input_image (str): A string representing the path of the input image.
        input_image (numpy.ndarray): An array representing the input image.
        image_bw (numpy.ndarray): An array representing the binary image obtained after preprocessing.
    """

    def __init__(self):
        # Initialize attributes to store detected centers and module sizes
        self.centers = []
        self.module_size = []
        # Initialize attributes for input image path and processed images
        self.path_input_image = ""
        self.input_image = None
        self.image_bw = None

    # Method to read an image from a specified path and preprocess it
    def read_image(self, path):
        self.path_input_image = path
        # Read the image from the specified path
        self.input_image = cv2.imread(self.path_input_image, cv2.IMREAD_COLOR)
        # Check if the image is successfully loaded
        if self.input_image is None:
            print("Could not open or find the image")
            exit(-1)
        # Convert the image to grayscale
        self.image_bw = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding to convert the grayscale image into binary
        self.image_bw = cv2.adaptiveThreshold(
            self.image_bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 0
        )

    # Method to create a result image with detected patterns drawn
    def create_result_image(self):
        path_result_image = self.path_input_image.replace(".jpg", "_result.jpg")
        cv2.imwrite(path_result_image, self.input_image)

    # Method to detect finder patterns in the image
    def detector(self):
        # Find patterns in the image
        found = self.find()
        # If patterns are found, draw rectangles around them
        if found:
            self.draw_finder_patterns()
        return found

    # Method to get the center of a pattern
    def get_center(self, counter_state, end):
        return end - counter_state[4] - counter_state[3] - counter_state[2] / 2.0

    # Method to find patterns in the image
    def find(self):
        # Clear previously detected centers and module sizes
        self.centers.clear()
        self.module_size.clear()

        counter_state = [0] * 5
        current_state = 0

        # Iterate through the image to find patterns
        for row in range(self.image_bw.shape[0]):
            counter_state = [0] * 5
            current_state = 0
            for col in range(self.image_bw.shape[1]):
                if self.image_bw[row, col] < 128:  # black pixel
                    if current_state & 0x1 == 1:
                        current_state += 1
                    counter_state[current_state] += 1
                else:  # white pixel
                    if current_state & 0x1 == 1:
                        counter_state[current_state] += 1
                    else:
                        if current_state == 4:
                            if not (
                                self.check_ratio(counter_state)
                                and self.handle_possible_center(counter_state, row, col)
                            ):
                                current_state = 3
                                counter_state[0:3] = counter_state[2:5]
                                counter_state[3:5] = [1, 0]
                                continue
                            current_state = 0
                            counter_state = [0] * 5
                        else:
                            current_state += 1
                            counter_state[current_state] += 1
        return len(self.centers) > 0

    # Method to check the ratio of black and white modules in a pattern
    def check_ratio(self, state_count):
        total_width = sum(state_count)
        if total_width < 7:
            return False

        width = total_width / 7.0
        dispersion = width / 2

        return (
            abs(width - state_count[0]) < dispersion
            and abs(width - state_count[1]) < dispersion
            and abs(3 * width - state_count[2]) < 3 * dispersion
            and abs(width - state_count[3]) < dispersion
            and abs(width - state_count[4]) < dispersion
        )

    # Method to handle a possible center of a pattern
    def handle_possible_center(self, state_count, row, col):
        total_state = sum(state_count)

        center_col = int(self.get_center(state_count, col))

        center_row = int(
            self.check_vertical(row, center_col, state_count[2], total_state)
        )
        if center_row == -1.0:
            return False

        center_col = int(
            self.check_horizontal(center_row, center_col, state_count[2], total_state)
        )
        if center_col == -1.0:
            return False

        if not self.check_diagonal(center_row, center_col, state_count[2], total_state):
            return False

        new_center = np.array([center_col, center_row])
        new_module_size = total_state / 7.0

        found = False
        for i, point in enumerate(self.centers):
            diff = point - new_center
            distance = np.sqrt(np.dot(diff, diff))
            if distance < 10:
                self.centers[i] = (point + new_center) / 2.0
                self.module_size[i] = (self.module_size[i] + new_module_size) / 2.0
                found = True
                break

        if not found:
            self.centers.append(new_center)
            self.module_size.append(new_module_size)

        return False

    # Method to check vertical continuity of a pattern
    def check_vertical(self, start_row, center_col, central_count, state_count_total):
        counter_state = [0] * 5
        row = start_row

        while row >= 0 and self.image_bw[row, center_col] < 128:
            counter_state[2] += 1
            row -= 1

        if row < 0:
            return -1.0

        while (
            row >= 0
            and self.image_bw[row, center_col] >= 128
            and counter_state[1] < central_count
        ):
            counter_state[1] += 1
            row -= 1

        if row < 0 or counter_state[1] >= central_count:
            return -1.0

        while (
            row >= 0
            and self.image_bw[row, center_col] < 128
            and counter_state[0] < central_count
        ):
            counter_state[0] += 1
            row -= 1

        if row < 0 or counter_state[0] >= central_count:
            return -1.0

        row = start_row + 1
        while row < self.image_bw.shape[0] and self.image_bw[row, center_col] < 128:
            counter_state[2] += 1
            row += 1

        if row == self.image_bw.shape[0]:
            return -1.0

        while (
            row < self.image_bw.shape[0]
            and self.image_bw[row, center_col] >= 128
            and counter_state[3] < central_count
        ):
            counter_state[3] += 1
            row += 1

        if row == self.image_bw.shape[0] or counter_state[3] >= state_count_total:
            return -1.0

        while (
            row < self.image_bw.shape[0]
            and self.image_bw[row, center_col] < 128
            and counter_state[4] < central_count
        ):
            counter_state[4] += 1
            row += 1

        if row == self.image_bw.shape[0] or counter_state[4] >= central_count:
            return -1.0

        counter_state_total = sum(counter_state)
        if 5 * abs(counter_state_total - state_count_total) >= 2 * state_count_total:
            return -1.0

        return (
            self.get_center(counter_state, row)
            if self.check_ratio(counter_state)
            else -1.0
        )

    # Method to check horizontal continuity of a pattern
    def check_horizontal(self, center_row, start_col, center_count, state_count_total):
        counter_state = [0] * 5
        col = start_col
        ptr = self.image_bw[center_row, :]

        while col >= 0 and ptr[col] < 128:
            counter_state[2] += 1
            col -= 1
        if col < 0:
            return -1.0

        while col >= 0 and ptr[col] >= 128 and counter_state[1] < center_count:
            counter_state[1] += 1
            col -= 1
        if col < 0 or counter_state[1] == center_count:
            return -1.0

        while col >= 0 and ptr[col] < 128 and counter_state[0] < center_count:
            counter_state[0] += 1
            col -= 1
        if col < 0 or counter_state[0] == center_count:
            return -1.0

        col = start_col + 1
        while col < self.image_bw.shape[1] and ptr[col] < 128:
            counter_state[2] += 1
            col += 1
        if col == self.image_bw.shape[1]:
            return -1.0

        while (
            col < self.image_bw.shape[1]
            and ptr[col] >= 128
            and counter_state[3] < center_count
        ):
            counter_state[3] += 1
            col += 1
        if col == self.image_bw.shape[1] or counter_state[3] == center_count:
            return -1.0

        while (
            col < self.image_bw.shape[1]
            and ptr[col] < 128
            and counter_state[4] < center_count
        ):
            counter_state[4] += 1
            col += 1
        if col == self.image_bw.shape[1] or counter_state[4] == center_count:
            return -1.0

        counter_state_total = sum(counter_state)
        if 5 * abs(state_count_total - counter_state_total) >= state_count_total:
            return -1.0

        return (
            self.get_center(counter_state, col)
            if self.check_ratio(counter_state)
            else -1.0
        )

    # Method to check diagonal continuity of a pattern
    def check_diagonal(self, center_row, center_col, max_count, state_count_total):
        state_count = [0] * 5
        i = 0
        while (
            center_row >= i
            and center_col >= i
            and self.image_bw[center_row - i, center_col - i] < 128
        ):
            state_count[2] += 1
            i += 1
        if center_row < i or center_col < i:
            return False

        while (
            center_row >= i
            and center_col >= i
            and self.image_bw[center_row - i, center_col - i] >= 128
            and state_count[1] <= max_count
        ):
            state_count[1] += 1
            i += 1
        if center_row < i or center_col < i or state_count[1] > max_count:
            return False

        while (
            center_row >= i
            and center_col >= i
            and self.image_bw[center_row - i, center_col - i] < 128
            and state_count[0] <= max_count
        ):
            state_count[0] += 1
            i += 1
        if state_count[0] > max_count:
            return False

        i = 1
        while (
            center_row + i < self.image_bw.shape[0]
            and center_col + i < self.image_bw.shape[1]
            and self.image_bw[center_row + i, center_col + i] < 128
        ):
            state_count[2] += 1
            i += 1
        if (
            center_row + i >= self.image_bw.shape[0]
            or center_col + i >= self.image_bw.shape[1]
        ):
            return False

        while (
            center_row + i < self.image_bw.shape[0]
            and center_col + i < self.image_bw.shape[1]
            and self.image_bw[center_row + i, center_col + i] >= 128
            and state_count[3] < max_count
        ):
            state_count[3] += 1
            i += 1
        if (
            center_row + i >= self.image_bw.shape[0]
            or center_col + i >= self.image_bw.shape[1]
            or state_count[3] > max_count
        ):
            return False

        while (
            center_row + i < self.image_bw.shape[0]
            and center_col + i < self.image_bw.shape[1]
            and self.image_bw[center_row + i, center_col + i] < 128
            and state_count[4] < max_count
        ):
            state_count[4] += 1
            i += 1
        if (
            center_row + i >= self.image_bw.shape[0]
            or center_col + i >= self.image_bw.shape[1]
            or state_count[4] > max_count
        ):
            return False

        state_count_total_new = sum(state_count)
        return abs(
            state_count_total - state_count_total_new
        ) < 2 * state_count_total and self.check_ratio(state_count)

    # Method to draw rectangles around detected finder patterns
    def draw_finder_patterns(self):
        if len(self.centers) == 0:
            return

        for i, pt in enumerate(self.centers):
            diff = self.module_size[i] * 3.5
            point1 = (int(pt[0] - diff), int(pt[1] - diff))
            point2 = (int(pt[0] + diff), int(pt[1] + diff))
            cv2.rectangle(self.input_image, point1, point2, (0, 0, 255), 1)


if __name__ == "__main__":
    # Create an instance of FinderPatternDetector class
    detector = FinderPatternDetector()

    # Specify the path of the input image
    path = "./qr1.jpg"

    # Read the input image
    detector.read_image(path)

    # Detect finder patterns in the image
    detector.detector()

    # Create a result image with detected patterns drawn
    detector.create_result_image()
