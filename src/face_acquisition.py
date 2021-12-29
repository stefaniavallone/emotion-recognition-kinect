import argparse
from face_recognition.dataset_acquisition import capture_photo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to acquire people photos for face recognition dataset.')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera to use')
    parser.add_argument('--name', type=str,
                        help='Name of the person')
    parser.add_argument('--num_photos', type=int,
                        help='Num of photos to take (default=30)', default=30)
    parser.add_argument('--data_dir', type=str,
                        help='Where to place photos', default="../resources/data/faces")
    args = parser.parse_args()

    capture_photo(base_path=args.data_dir,
                  name=args.name,
                  num_photos=args.num_photos,
                  camera_source=args.camera)
