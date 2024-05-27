import os

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from graphs.model_accuracy_plot import plot_accuracy
from graphs.model_loss_plot import plot_loss

train_dir = '../data/dataset/train'
validation_dir = '../data/dataset/validation'
batch_size = 32
img_width, img_height = 180, 180

# Use ImageDataGenerator to preprocess and augment training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale validation data (no augmentation)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Create data generators for training and validation sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'  # 'binary' for binary classification (e.g., 'occupied' vs 'empty')
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 'binary_crossentropy' for binary classification
              metrics=['accuracy'])

# Display the model summary
model.summary()

epochs = 100

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size
)

# Define the directory path to save the model
save_dir = '../data/results/'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Save the model to the specified path
model.save('../data/results/trained_model.h5', save_format='h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plot_accuracy(epochs_range, acc, val_acc)
plot_loss(epochs_range, loss, val_loss)

# # Plot training and validation accuracy
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.text(0.5, 0.5, f'Training Accuracy: {acc[-1]:.4f}\nValidation Accuracy: {val_acc[-1]:.4f}',
#          horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
# plt.savefig(os.path.join('data/dataset/results/', 'accuracy_plot.png'))
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.text(0.5, 0.5, f'Training Loss: {loss[-1]:.4f}\nValidation Loss: {val_loss[-1]:.4f}',
#          horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
# plt.savefig(os.path.join('data/dataset/results/', 'loss_plot.png'))
# plt.show()
