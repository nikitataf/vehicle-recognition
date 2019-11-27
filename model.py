import numpy as np
import glob
import os
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


def model_simple(args):
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


def model_ResNet(args):
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', pooling='max')

    model = Sequential()
    model.add(base_model)
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(args.num_classes, activation='softmax'))

    return model


def model_MobileNet(args):
    base_model = tf.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', pooling='max')

    model = Sequential()
    model.add(base_model)
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(args.num_classes, activation='softmax'))

    return model


def model_NASNet(args):
    base_model = tf.keras.applications.nasnet.NASNetMobile(include_top=False, weights='imagenet', pooling='max')

    model = Sequential()
    model.add(base_model)
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(args.num_classes, activation='softmax'))

    return model


def train(args, train_generator, validation_generator):

    # Construct a model
    model = model_simple(args)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Stop training when a monitored quantity has stopped improving
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    # Save the best model
    file_path = 'model/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    best_model = tf.keras.callbacks.ModelCheckpoint(file_path, save_best_only=True, monitor='val_loss')
    # Save the weights
    file_path = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    model_weights = tf.keras.callbacks.ModelCheckpoint(file_path, save_best_only=True, save_weights_only=True,
                                                       monitor='loss', mode='auto', period=1, verbose=0)

    # Train model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=22443 // args.batch_size,
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=5602 // args.batch_size,
        callbacks=[earlyStopping, best_model, model_weights]
    )

    weight_files = glob.glob(os.path.join(os.getcwd(), 'weights/*'))
    weight_file = max(weight_files, key=os.path.getctime)  # most recent file

    # plot_accuracy(args, history)
