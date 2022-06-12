from keras.metrics import CategoricalAccuracy
from keras.models import load_model
from sklearn.metrics import confusion_matrix

from src.graphics import plot_confusion_matrix
from src.normalization import normalize_pixels_x, categorize_y
from src.utils import load_cifar10, load_cifar10_labels

if __name__ == '__main__':
    # Load the test CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    labels = load_cifar10_labels()

    # Preprocessing
    x_train, x_test = normalize_pixels_x(x_train, x_test)
    y_train, y_test = categorize_y(y_train, y_test)

    # Load the trained models (one or more depending on task and bonus)
    # for example
    model_task1 = load_model('./deliverable/nn_task1.h5')

    # Predict on the given samples
    y_pred_task1 = model_task1.predict(x_test)

    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_task1.argmax(axis=1))
    plot_confusion_matrix(matrix, labels)

    # Evaluate the missclassification error on the test set
    assert y_test.shape == y_pred_task1.shape
    ca = CategoricalAccuracy()
    ca.update_state(y_test, y_pred_task1)
    acc1 = ca.result().numpy()
    print("Accuracy model task 1:", acc1)
