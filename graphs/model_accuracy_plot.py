import os

import matplotlib.pyplot as plt


def plot_accuracy(epochs_range, acc, val_acc):
    # Plot training and validation accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.text(0.02, 0.95, f'Training Accuracy: {acc[-1]:.4f}\nValidation Accuracy: {val_acc[-1]:.4f}',
             color='black', fontsize=10, transform=plt.gca().transAxes, ha='left')
    plt.savefig(os.path.join('../data/images/graphs', 'accuracy_plot_1.png'))
    plt.show()
