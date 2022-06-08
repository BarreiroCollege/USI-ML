import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn


def plot_history(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)

    # Auxiliary info and funcs
    best_epoch_loss = np.argmin(history.history['val_loss'])
    best_epoch_acc = np.argmax(history.history['val_accuracy'])
    epochs = history.epoch

    # Plot training & validation accuracy values
    plt.plot(smooth(history.history['loss'], epochs), c='C0', alpha=0.7, lw=3)
    plt.plot(smooth(history.history['val_loss'], epochs), c='C1', alpha=0.7, lw=3)
    plt.axvline(best_epoch_loss, label='best_epoch', c='k', ls='--', alpha=0.3)
    # Empirical values
    plt.plot(history.history['loss'], label='train_loss', c='C0')
    plt.plot(history.history['val_loss'], label='val_loss', c='C1')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()

    plt.subplot(122)
    # Plot training & validation accuracy values
    plt.plot(smooth(history.history['accuracy'], epochs), c='C0', alpha=0.7, lw=3)
    plt.plot(smooth(history.history['val_accuracy'], epochs), c='C1', alpha=0.7, lw=3)
    plt.axvline(best_epoch_acc, label='best_epoch', c='k', ls='--', alpha=0.3)
    # Empirical values
    plt.plot(history.history['accuracy'], label='train_accuracy', c='C0')
    plt.plot(history.history['val_accuracy'], label='val_accuracy', c='C1')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()

    plt.show()


def smooth(y, epochs):
    return np.polyval(np.polyfit(epochs, y, deg=5), epochs)


def plot_confusion_matrix(matrix, labels):
    df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.show()
