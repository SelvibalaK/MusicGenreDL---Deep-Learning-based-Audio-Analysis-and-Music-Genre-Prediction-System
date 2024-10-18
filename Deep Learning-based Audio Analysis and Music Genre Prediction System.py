#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2
import random
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt


# In[23]:


# Load data and preprocess
def load_and_trim_audio(file_path):
    audio, sr = librosa.load(file_path)
    audio, _ = librosa.effects.trim(audio)
    return audio, sr


# In[24]:


def load_and_preprocess_data(audio_dir, genres):
    audio_files = []
    labels = []
    for genre in genres:
        for file in os.listdir(f'{audio_dir}/{genre}'):
            audio, sr = load_and_trim_audio(f'{audio_dir}/{genre}/{file}')
            mel_spectrogram = create_mel_spectrogram(audio, sr)
            audio_files.append(mel_spectrogram)
            labels.append(genre)
    return audio_files, labels


# In[25]:


def load_image_data(image_dir, genres):
    image_files = []
    labels = []
    for genre in genres:
        genre_dir = os.path.join(image_dir, genre)
        for file in os.listdir(genre_dir):
            if file.endswith('.png'):  # Assuming images are in PNG format
                image_path = os.path.join(genre_dir, file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                image = cv2.resize(image, (224, 224))  # Resize to match VGG16 input
                image = image / 255.0  # Normalize pixel values
                image_files.append(image)
                labels.append(genre)
    return image_files, labels


# In[28]:


# Convert to numpy array
X = np.array(image_files, dtype=np.float32)  # Ensure float32 data type
y = np.array(labels)


# In[30]:


# Convert labels to numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)


# In[31]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])).reshape(X_test.shape)


# In[33]:


# Define custom CNN model architecture
input_shape = X_train.shape[1:]
inputs = Input(shape=input_shape)
custom_model = Sequential([
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
  # Added layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),  # Increased neurons
    Dropout(0.2),  # Added dropout for regularization
    Dense(10, activation='softmax')
])


# In[34]:


# Compile custom model
custom_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[35]:


# Train custom model
custom_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# In[36]:


# Define transfer learning model architecture using InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])


# In[37]:


# Freeze initial layers
for layer in base_model.layers[:10]:
    layer.trainable = False


# In[38]:


# Add custom layers on top
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(10, activation='softmax')(x)
transfer_model = Model(inputs=base_model.input, outputs=x)


# In[46]:


# Compile transfer model
transfer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[47]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)  
print(y_test.shape)  


# In[48]:


# Train transfer model
transfer_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)


# In[50]:


# Make predictions on test set using custom model
y_pred_custom = custom_model.predict(X_test)


# In[51]:


# Make predictions on test set using transfer model
y_pred_transfer = transfer_model.predict(X_test)


# In[52]:


# Convert predictions to probabilities
probabilities_custom = tf.nn.softmax(y_pred_custom).numpy()
probabilities_transfer = tf.nn.softmax(y_pred_transfer).numpy()


# In[53]:


# Get genre with highest probability
predicted_genres_custom = np.argmax(probabilities_custom, axis=1)
predicted_genres_transfer = np.argmax(probabilities_transfer, axis=1)


# In[54]:


# Evaluate models
accuracy_custom = accuracy_score(y_test, predicted_genres_custom)
accuracy_transfer = accuracy_score(y_test, predicted_genres_transfer)
print(f'Custom model accuracy: {accuracy_custom:.3f}')
print(f'Transfer model accuracy: {accuracy_transfer:.3f}')
print(classification_report(y_test, predicted_genres_custom))
print(classification_report(y_test, predicted_genres_transfer))
print(confusion_matrix(y_test, predicted_genres_custom))
print(confusion_matrix(y_test, predicted_genres_transfer))


# In[70]:


def choose_random_sample(X_test, y_test):
    index = np.random.randint(len(X_test))
    sample = X_test[index]
    true_genre = y_test[index]
    return sample, true_genre


# In[71]:


# Define function to predict genre of new audio file using custom model
def predict_genre_custom(audio_file):
    audio, sr = load_and_trim_audio(audio_file)
    mel_spectrogram = create_mel_spectrogram(audio, sr)
    mel_spectrogram = mel_spectrogram.reshape((1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1))
    mel_spectrogram = scaler.transform(mel_spectrogram.reshape(-1, mel_spectrogram.shape[1] * mel_spectrogram.shape[2])).reshape(mel_spectrogram.shape)
    prediction = custom_model.predict(mel_spectrogram)
    probabilities = tf.nn.softmax(prediction).numpy()
    predicted_genre = np.argmax(probabilities, axis=1)[0]
    return genres[predicted_genre], probabilities[0]


# In[72]:


# Define function to predict genre of new audio file using transfer model
def predict_genre_transfer(audio_file):
    audio, sr = load_and_trim_audio(audio_file)
    mel_spectrogram = create_mel_spectrogram(audio, sr)
    mel_spectrogram = mel_spectrogram.reshape((1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1))
    mel_spectrogram = scaler.transform(mel_spectrogram.reshape(-1, mel_spectrogram.shape[1] * mel_spectrogram.shape[2])).reshape(mel_spectrogram.shape)
    prediction = transfer_model.predict(mel_spectrogram)
    probabilities = tf.nn.softmax(prediction).numpy()
    predicted_genre = np.argmax(probabilities, axis=1)[0]
    return genres[predicted_genre], probabilities[0]


# In[73]:


def predict_and_display(model, sample):
    prediction = model.predict(np.expand_dims(sample, axis=0))
    probabilities = tf.nn.softmax(prediction).numpy()[0]
    predicted_genre = np.argmax(probabilities)
    return predicted_genre, probabilities


# In[75]:


def display_mel_spectrogram(audio_file_path):
    audio, sr = librosa.load(audio_file_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout() 
    plt.show()


# In[76]:


def play_audio(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    ipd.Audio(audio.raw_data, rate=audio.frame_rate)


# In[101]:


def main():
    while True:
        choice = input("Choose: 1. Random sample, 2. Insert audio file, 3. Insert Mel Spectrogram image: ")
        
        if choice == "2":
            audio_file = input("Enter audio file path: ")
            try:
                audio, sr = librosa.load(audio_file)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
                mel_spectrogram_resized = cv2.resize(mel_spectrogram, (224, 224))
                mel_spectrogram_resized = mel_spectrogram_resized.astype(np.float32) / 255.0
                mel_spectrogram_resized = np.stack((mel_spectrogram_resized,) * 3, axis=-1)
                features = np.expand_dims(mel_spectrogram_resized, axis=0)

                # Custom model prediction
                prediction_custom = custom_model.predict(features)
                probabilities_custom = tf.nn.softmax(prediction_custom).numpy()[0]
                predicted_genre_custom = np.argmax(probabilities_custom)

                # Transfer model prediction
                prediction_transfer = transfer_model.predict(features)
                probabilities_transfer = tf.nn.softmax(prediction_transfer).numpy()[0]
                predicted_genre_transfer = np.argmax(probabilities_transfer)

                print(f"Custom model predicted genre: {label_encoder.inverse_transform([predicted_genre_custom])[0]}")
                print(f"Custom model probabilities: {probabilities_custom}")
                print(f"Transfer model predicted genre: {label_encoder.inverse_transform([predicted_genre_transfer])[0]}")
                print(f"Transfer model probabilities: {probabilities_transfer}")

                display_mel_spectrogram(audio, sr)
                play_audio(audio_file)

            except Exception as e:
                print(f"Error processing audio file: {e}")

        else:
            print("Invalid choice. Please choose 1, 2, or 3.")

        continue_loop = input("Do you want to continue? (y/n): ")
        if continue_loop.lower() != "y":
            break


# In[ ]:


if __name__ == "__main__":
    main()


# In[97]:


if __name__ == "__main__":
    main()


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




