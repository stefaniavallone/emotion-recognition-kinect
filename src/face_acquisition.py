import argparse
from face_recognition.dataset_acquisition import capture_photo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture photo to train the face recognizer.')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera to use')
    parser.add_argument('--name', type=str,
                        help='Name of the person')
    parser.add_argument('--num_photos', type=int,
                        help='Name of the person', default=30)
    args = parser.parse_args()

    capture_photo(base_path="../resources/data/faces",
                  name=args.name,
                  num_photos=args.num_photos,
                  camera_source=args.camera)
