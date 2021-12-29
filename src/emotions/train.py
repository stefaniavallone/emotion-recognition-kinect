import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


def train_model(model, train_dir,
                batch_size=32,
                num_epochs=100,
                verbose=0):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=True,
        subset='training',
        class_mode='categorical')

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=False,
        subset='validation',
        class_mode='categorical')

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=20, verbose=1, mode='auto',
                                   baseline=None, restore_best_weights=True)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        shuffle=True,
        verbose=verbose,
        callbacks=[early_stopping])
    if verbose >= 1:
        plot_model_history(history)
    return model


def plot_model_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(history.history['accuracy']) + 1),
                history.history['accuracy'])
    axs[0].plot(range(1, len(history.history['val_accuracy']) + 1),
                history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(history.history['loss']) + 1),
                history.history['loss'])
    axs[1].plot(range(1, len(history.history['val_loss']) + 1),
                history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('train.png')
    plt.show()
