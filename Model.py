import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Load and preprocess the dataset
data = pd.read_csv("C:/Users/kshit/Desktop/Handwriting Recognition/data/data.csv")
dataset = data.values
np.random.shuffle(dataset)

X = dataset[:, :1024] / 255.0
Y = dataset[:, 1024]

# Split the data into training and testing sets
X_train, X_test = X[:70000], X[70000:72001]
Y_train, Y_test = Y[:70000], Y[70000:72001]

# One-hot encode the labels
Y_train = to_categorical(Y_train, num_classes=37)
Y_test = to_categorical(Y_test, num_classes=37)

print(f"Number of training examples = {X_train.shape[0]}")
print(f"Number of test examples = {X_test.shape[0]}")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")

# Reshape the input data to fit the model's requirements
image_x, image_y = 32, 32
X_train = X_train.reshape(-1, image_x, image_y, 1)
X_test = X_test.reshape(-1, image_x, image_y, 1)


# Define the model
def build_model(image_x, image_y):
    num_classes = 37
    model = Sequential([
        Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(image_x, image_y, 1)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Conv2D(64, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'),
        Flatten(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define the model checkpoint
    checkpoint = ModelCheckpoint("devanagari.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    return model, [checkpoint]


# Build and train the model
model, callbacks_list = build_model(image_x, image_y)
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=8, batch_size=64, callbacks=callbacks_list)

# Evaluate the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Accuracy: {scores[1] * 100:.2f}%")

# Print model summary
print(model.summary())

# Save the model
model.save('devanagari.keras')
