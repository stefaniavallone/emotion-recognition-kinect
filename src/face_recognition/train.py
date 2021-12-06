import os
import pickle

import cv2
import numpy as np


def train_recognizer(recognizer, train_dir):
	faces, labels, labels2names_map = get_faces_and_names(train_dir)
	recognizer.train(faces, labels)  # Train for facial recognition
	return recognizer, labels2names_map


def save_recognizer(recognizer, labels_name_map, recognizer_name, base_path=f'../resources/models/faces'):
	model_path = os.path.join(base_path, recognizer_name + ".yaml")
	recognizer.save(model_path)  # Dump the trained model
	print(f"Saved FaceRecognition model to disk: {model_path}")
	labels2name_map_path = os.path.join(base_path, recognizer_name + "_labels.bin")
	with open(labels2name_map_path, 'wb') as labels_name_map_file:
		pickle.dump(labels_name_map, labels_name_map_file,
					protocol=pickle.HIGHEST_PROTOCOL)  # Dump the label:name map
		print(f"Saved FaceRecognition labels to disk: {labels2name_map_path}")


def get_faces_and_names(path):
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png") | f.endswith(".jpg")]
	faces = []
	labels = []
	labels2names_map = dict()
	names2labels_map = dict()
	people_counter = 0
	for image_path in image_paths:
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		faces.append(image)
		name = os.path.split(image_path)[1].split(".")[0]
		if name not in names2labels_map:
			names2labels_map[name] = people_counter
			labels2names_map[people_counter] = name
			people_counter = people_counter + 1

		labels.append(names2labels_map[name])
	return faces, np.array(labels), labels2names_map
