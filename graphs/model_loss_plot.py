# Plot training and validation accuracy
import os

import matplotlib.pyplot as plt


def plot_loss(epochs_range, loss, val_loss):
    # Plot training and validation loss
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.text(0.02, 0.95, f'Training Loss: {loss[-1]:.4f}\nValidation Loss: {val_loss[-1]:.4f}',
             color='black', fontsize=10, transform=plt.gca().transAxes, ha='left')
    plt.savefig(os.path.join('../data/images/graphs', 'loss_plot.png'))
    plt.show()
