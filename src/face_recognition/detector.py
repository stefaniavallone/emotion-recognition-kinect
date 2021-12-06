import cv2


class FaceDetectorModel:

    def __init__(self, cascade_path=f"../resources/models/faces/haarcascade_frontalface_default.xml",
                 scale_factor=1.15,
                 min_neighbors=5,
                 min_size=(30, 30),
                 face_resolution=(150, 150),
                 cascade_flags=cv2.CASCADE_SCALE_IMAGE):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.face_resolution = face_resolution
        self.cascade_flags = cascade_flags
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def predict(self, image):
        faces = self.face_cascade.detectMultiScale(image,
                                                   self.scale_factor,
                                                   self.min_neighbors,
                                                   self.cascade_flags,
                                                   self.min_size)
        return [{"pos": (x, y, w, h),
                 "img": cv2.resize(image[y: y + h, x: x + w],
                                   self.face_resolution)}
                for (x, y, w, h) in faces]
