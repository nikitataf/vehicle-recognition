import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


def plot_accuracy(args, history):
    # Plot training & validation accuracy and lossvalues
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(args.epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.grid()
    plt.title('Training and Validation Loss')

    plt.show()


def model_1(args):
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(args.IMG_HEIGHT, args.IMG_WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(args.num_classes, activation='softmax')
    ])
    return model


def model_3(num_classes):
    #TODO Need to figure out how to use it
    model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_tensor=None,
                     input_shape=(200, 200, 3), pooling=None, classes=num_classes)

    return model


def train(args, train_generator, validation_generator):

    # Construct a model
    model = model_1(args)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Stop training when a monitored quntity has stopped improving
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=300, verbose=0)
    # Save the best model
    best_model = tf.keras.callbacks.ModelCheckpoint('best_model.hdf5', save_best_only=True, monitor='val_loss')

    # Train model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=22443 // args.batch_size,
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=5602 // args.batch_size,
        callbacks=[earlyStopping, best_model]
    )

    # plot_accuracy(args, history)
