import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Corrected import

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

# Print number of samples in each directory
num_train_samples = train_generator.samples
num_val_samples = validation_generator.samples

print(f"Number of training samples: {num_train_samples}")
print(f"Number of validation samples: {num_val_samples}")

# Calculate steps per epoch
steps_per_epoch = num_train_samples // batch_size
validation_steps = num_val_samples // batch_size

# Define the model with Dropout and L2 Regularization
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
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),  # Dropout layer to prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 'binary_crossentropy' for binary classification
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

epochs = 100

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping]  # Add early stopping to the training process
)

# Define the directory path to save the model
save_dir = '../data/model/'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Save the model to the specified path
model.save('../data/model/trained_model.h5', save_format='h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))  # Use the length of the accuracy list to handle early stopping

plot_accuracy(epochs_range, acc, val_acc)
plot_loss(epochs_range, loss, val_loss)
