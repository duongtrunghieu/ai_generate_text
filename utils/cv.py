import numpy as np
import cv2
import base64

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return image


def process_image(image):
    gray_image = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 112, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    heights = [cv2.boundingRect(cnt)[3] for cnt in contours]
    max_height = max(heights)
    median_height = np.median(heights)

    max_height_threshold = median_height

    blank_image = np.zeros_like(gray_image)

    for i in range(len(contours)):
        cnt = contours[i]
        _, _, w, h = cv2.boundingRect(cnt)

        if max_height_threshold / max_height < 0.55:
            if h < max_height_threshold + 4 :
                cv2.drawContours(blank_image, contours, i, (255, 255, 255), 2)
        else:
            blank_image = gray_image
    # debug
    # cv2.imshow("blank_image", blank_image)
    # cv2.waitKey(0)
    return blank_image