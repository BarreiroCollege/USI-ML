from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense
from keras.optimizer_v2.rmsprop import RMSprop
from keras.utils.vis_utils import plot_model

from src.graphics import plot_history
from src.normalization import normalize_pixels_x, categorize_y
from src.settings import NUM_CLASSES
from src.utils import load_cifar10, save_keras_model


def define_model(input_shape=(32, 32, NUM_CLASSES)):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=NUM_CLASSES, activation='softmax'))
    model.compile(optimizer=RMSprop(learning_rate=0.003), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def fit_model(model, x_train, y_train):
    history = model.fit(
        x_train, y_train,
        epochs=500, batch_size=128,
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)],
        validation_split=0.2,
        shuffle=False, verbose=1
    )
    return history


def main():
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, x_test = normalize_pixels_x(x_train, x_test)
    y_train, y_test = categorize_y(y_train, y_test)

    model = define_model()
    # plot_model(model, to_file='./img/task2_model.png', show_shapes=True, show_layer_names=True)
    history = fit_model(model, x_train, y_train)
    plot_history(history)
    loss, acc = model.evaluate(x_test, y_test, verbose=1)

    save_keras_model(model, './deliverable/nn_task2.h5')

    print('--- --- ---')
    print('Loss: {} - Accuracy: {}'.format(loss, acc))


if __name__ == '__main__':
    main()
