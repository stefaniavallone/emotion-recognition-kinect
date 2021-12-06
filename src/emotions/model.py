import os

import keras2onnx
import onnx
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, \
    Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.python.keras.layers import BatchNormalization

labels = {0: "Angry", 1: "Disgusted",
          2: "Fearful", 3: "Happy",
          4: "Neutral", 5: "Sad", 6: "Surprised"}


def create_model(kernel_size=3, dropout=0.25, lr=0.0001, decay=1e-6):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu',
                     input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(256, kernel_size=kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr, decay=decay),
                  metrics=['accuracy',
                           Recall(name='recall'),
                           Precision(name='precision')])

    return model


def save_model(model, model_name, base_path=f'../resources/models/emotions'):
    model_path = os.path.join(base_path, f"{model_name}.h5")
    model.save_weights(model_path)
    print(f"Saved Keras model to disk: {model_path}.")
    onnx_model = convert_to_onnx(model, model_name)
    onnx_model_path = os.path.join(base_path, f"{model_name}.onnx")
    onnx.save_model(onnx_model, onnx_model_path)
    print(f"Saved ONNX model to disk: {onnx_model_path}")


def convert_to_onnx(model, model_name):
    onnx_model = keras2onnx.convert_keras(model, model_name)
    return onnx_model