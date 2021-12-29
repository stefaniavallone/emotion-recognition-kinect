import os

import keras2onnx
import onnx
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, \
    Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import BatchNormalization


labels = {0: "Angry", 1: "Disgusted",
          2: "Fearful", 3: "Happy",
          4: "Neutral", 5: "Sad", 6: "Surprised"}


def create_model(kernel_size=3,
                 l2=0.04370766874155845,
                 dropout=0.3106791934814161,
                 lr=6.454516989719096e-05,
                 decay=4.461966074951546e-05):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu',
                     input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu', kernel_regularizer=regularizers.l2(l=l2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=kernel_size, activation='relu', kernel_regularizer=regularizers.l2(l=l2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=kernel_size, activation='relu', kernel_regularizer=regularizers.l2(l=l2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(dropout+0.25))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr, decay=decay),
                  metrics=['accuracy'])
    return model


def save_model(model, model_name, base_path=f'../resources/models/emotions'):
    model_path = os.path.join(base_path, f"{model_name}")
    model.save(model_path)
    print(f"Saved Tensorflow model to disk: {model_path}.")
    model.save_weights(model_path + ".h5")
    print(f"Saved Keras model to disk: {model_path}.h5.")
    onnx_model = convert_to_onnx(model, model_name)
    onnx_model_path = os.path.join(base_path, f"{model_name}.onnx")
    onnx.save_model(onnx_model, onnx_model_path)
    print(f"Saved ONNX model to disk: {onnx_model_path}")


def convert_to_onnx(model, model_name):
    onnx_model = keras2onnx.convert_keras(model, model_name)
    return onnx_model