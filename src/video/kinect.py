import freenect
import cv2
import numpy as np


from video.process_frame import process_frame


def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


# function to get depth image from video
def get_depth():
    array, _ = freenect.sync_get_depth()
    min_depth = 0
    max_depth = array.max()
    array = ((array.astype(np.float32) - min_depth) / (max_depth - min_depth)) * 255
    return array.astype(np.uint8)


def clean_and_shift_image(frame, k=(3, 3)):
    # Shift part (remove or change based on your Kinect calibration)
    M = np.float32([
        [1, 0, -35],
        [0, 1, 30]
    ])
    frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderValue=255)

    # Classic CV cleaning
    kernel = np.ones(k, np.uint8)
    img_erosion = cv2.erode(frame, kernel, iterations=3)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=3)

    # Filtering by distance with Binarization
    th, depth_binary = cv2.threshold(img_dilation, 100, 255, cv2.THRESH_BINARY_INV)
    return depth_binary


def process_and_display(confirmation_window_handler, face_recognition_model, emotion_model):
    while 1:
        frame = get_video()
        depth = get_depth()

        clean_depth = clean_and_shift_image(depth, k=(7, 7))

        filtered_frame = frame.copy()
        filtered_frame[np.where(clean_depth == 0)] = 0

        # Process frame
        result_frame = process_frame(filtered_frame, confirmation_window_handler,
                                     face_recognition_model, emotion_model)
        frame_and_result_frame = np.concatenate((frame, result_frame), axis=1)

        # quit program when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('Result', cv2.resize(frame_and_result_frame,
                                        (int(frame_and_result_frame.shape[1]*1.25),
                                         int(frame_and_result_frame.shape[0]*1.25)),
                                        interpolation=cv2.INTER_AREA))
    cv2.destroyAllWindows()
