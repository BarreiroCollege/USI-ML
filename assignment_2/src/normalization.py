from keras.utils.np_utils import to_categorical

from src.settings import NUM_CLASSES


def normalize_pixels_x(train, test):
    x_train = train.astype('float32') / 255.0
    x_test = test.astype('float32') / 255.0
    return x_train, x_test


def categorize_y(train, test, num_classes=NUM_CLASSES):
    y_train = to_categorical(train, num_classes)
    y_test = to_categorical(test, num_classes)
    return y_train, y_test
