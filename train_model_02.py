# train_model.py
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.applications.vgg16 import VGG16
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint ,  ReduceLROnPlateau

import matplotlib.pyplot as plt

# Define image dimensions and batch size
img_size = (150, 150)
batch_size = 32
test_data="data/seg_test/seg_test"
train_data="data/seg_train/seg_train"

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

# No augmentation for validation
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(train_data,
                                                    target_size=img_size,
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory(train_data,
                                                         target_size=img_size,
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         subset='validation')

test_generator = test_datagen.flow_from_directory(test_data,
                                                  target_size=img_size,
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=False)

### Building the model
# Define input shape
input_shape = (150, 150, 3)
inputs = Input(shape=input_shape)

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

# Freeze the base model initially
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze the top layers (last 4 convolutional blocks)
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(6, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', verbose=1)

# Train the model with the defined callbacks
history = model.fit(train_generator,
                    epochs=50,
                    validation_data=validation_generator,
                    callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Evaluate the model
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Time')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Time')

plt.show()

# Save the trained model
model_name="image_classifier_model_v2.keras"
model.save(f'models/{model_name}')
print(f"Model saved to models/{model_name}")