import cv2
from video.process_frame import process_frame


def process_and_display(camera_source=0,
                        confirmation_window_handler=None,
                        face_recognition_model=None,
                        emotion_model=None):
    cv2.ocl.setUseOpenCL(False)
    video_capture = cv2.VideoCapture(camera_source)

    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = process_frame(frame, confirmation_window_handler,
                              face_recognition_model, emotion_model)

        cv2.imshow('Video', cv2.resize(frame, (1280, 720),
                                       interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
