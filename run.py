import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Activation, Flatten, Dropout, Dense, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from keras import regularizers
from keras.applications.resnet50 import ResNet50


def plot_accuracy(history):
    # Plot training & validation accuracy values
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig.savefig('figures/accuracy.png', dpi=300)

    # Plot training & validation loss values
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig.savefig('figures/loss.png', dpi=300)


def model_1():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='softmax'))
    model.summary()

    return model


# Model 2
def model_2(input_shape, weight_decay):

    model = Sequential()

    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regularizers.l2(weight_decay) ))
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dropout(0.7))
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dropout(0.7))
    model.add(Conv2D(192, (3, 3), padding = 'same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1),padding='valid', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(9, (1, 1), padding='valid', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(rate=0.25))
    model.add(Dense(22))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.summary()
    return model


def model_3(num_classes):
    model = ResNet50(include_top=True, weights=None, input_tensor=None,
                     input_shape=(200, 200, 3), pooling=None, classes=num_classes)

    return model


def train(X, y, params):
    # One-Hot encoding
    y = np.array(to_categorical(y, params.num_classes))

    # Split to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

    input_shape = (200, 200, 3)

    model = model_1()
    # model = model_2(input_shape, params.weight_decay)
    # model_3(params.num_classes)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=300, verbose=0)
    best_model = ModelCheckpoint('best_model.hdf5', save_best_only=True, monitor='val_loss')

    # Train model
    history = model.fit(x=X_train, y=y_train, epochs=params.eps, batch_size=params.batch_size, shuffle=False,
                        validation_data=(X_test, y_test),
                        callbacks=[earlyStopping, best_model])
    print('Training done!')

    # Evaluate model
    print(model.evaluate(X_test, y_test))
    plot_accuracy(history)
