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

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(args.num_classes, activation='softmax'))
    model.summary()

    return model


def model_2(args):
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(args.IMG_HEIGHT, args.IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(args.num_classes, activation='sigmoid')
    ])
    return model


def model_3(num_classes):
    model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_tensor=None,
                     input_shape=(200, 200, 3), pooling=None, classes=num_classes)

    return model


def train_old(X, y, params):
    # One-Hot encoding
    y = np.array(tf.keras.utils.to_categorical(y, params.num_classes))

    # Split to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

    input_shape = (200, 200, 3)

    model = model_1(params)
    # model = model_2(input_shape, params.weight_decay)
    # model_3(params.num_classes)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=300, verbose=0)
    best_model = tf.keras.callbacks.ModelCheckpoint('best_model.hdf5', save_best_only=True, monitor='val_loss')

    # Train model
    history = model.fit(x=X_train, y=y_train, epochs=params.epochs, batch_size=params.batch_size, shuffle=False,
                        validation_data=(X_test, y_test),
                        callbacks=[earlyStopping, best_model])
    print('Training done!')

    # Evaluate model
    # print(model.evaluate(X_test, y_test))
    # plot_accuracy(history)


def train(args, train_generator, validation_generator):

    model = model_2(args)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=300, verbose=0)
    best_model = tf.keras.callbacks.ModelCheckpoint('best_model.hdf5', save_best_only=True, monitor='val_loss')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=334 // args.batch_size,
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=334 // args.batch_size,
        callbacks=[earlyStopping, best_model]
    )

    plot_accuracy(args, history)
