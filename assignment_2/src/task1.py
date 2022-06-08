import operator

from keras.callbacks import EarlyStopping
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.optimizer_v2.rmsprop import RMSprop
from keras.utils.vis_utils import plot_model

from src.graphics import plot_history
from src.normalization import normalize_pixels_x, categorize_y
from src.settings import NUM_CLASSES
from src.utils import load_cifar10, save_keras_model


class GridCombination:
    model, history = None, None
    loss, acc = None, None
    learning_rate, neurons = None, None

    def __init__(self, learning_rate, neurons):
        self.learning_rate = learning_rate
        self.neurons = neurons


def define_model(learning_rate=0.003, neurons=8, input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=neurons, activation='tanh'))
    model.add(Dense(units=NUM_CLASSES, activation='softmax'))
    model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def fit(model, x_train, y_train, verbose=0):
    history = model.fit(
        x_train, y_train,
        epochs=500, batch_size=128,
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)],
        validation_split=0.2,
        shuffle=False, verbose=verbose
    )
    return history


def main():
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, x_test = normalize_pixels_x(x_train, x_test)
    y_train, y_test = categorize_y(y_train, y_test)

    model = define_model()
    # plot_model(model, to_file='./img/task1_model.png', show_shapes=True, show_layer_names=True)
    history = fit(model, x_train, y_train, verbose=1)
    plot_history(history)
    loss, acc = model.evaluate(x_test, y_test, verbose=1)

    save_keras_model(model, './deliverable/nn_task1.h5')

    # === BONUS === #

    learning_rates = [0.01, 0.0001]
    neurons = [16, 64]
    models = []

    for lr in learning_rates:
        for neuron in neurons:
            print(lr, neuron)
            comb = GridCombination(lr, neuron)
            comb.model = define_model(learning_rate=lr, neurons=neuron)
            comb.history = fit(comb.model, x_train, y_train)
            comb.loss, comb.acc = comb.model.evaluate(x_test, y_test, verbose=1)
            models.append(comb)

    best = max(models, key=operator.attrgetter('acc'))

    print('--- --- ---')
    print('Default Model')
    print('Loss: {} - Accuracy: {}'.format(loss, acc))
    print('---')
    print('Best Grid Model (accuracy)')
    print('Learning Rate: {} - Neurons: {}'.format(best.learning_rate, best.neurons))
    print()
    for m in models:
        print('Grid(lr={}, n={}) -> Loss: {} - Accuracy: {}'.format(m.learning_rate, m.neurons, m.loss, m.acc))


if __name__ == '__main__':
    main()
