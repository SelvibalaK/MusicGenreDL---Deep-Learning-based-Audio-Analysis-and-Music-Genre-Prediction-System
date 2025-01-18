# MusicGenreDL: Deep Learning-Based Audio Analysis and Music Genre Prediction System
Welcome to MusicGenreDL, a deep learning-based audio analysis and music genre prediction system! This project represents my first hands-on exploration into the fascinating intersection of deep learning and audio processing. It aims to analyze audio data and predict its corresponding music genre using state-of-the-art neural network architectures and audio feature extraction techniques.
Developed as part of an advanced learning initiative, the project serves as a foundational step for exploring and understanding the complex world of audio analysis and genre classification.

## About
This project implements a comprehensive music genre prediction system using deep learning and audio feature extraction techniques. The approach involves transforming raw audio data into meaningful visual and feature representations, which are then passed into a neural network for classification.
The system supports ten different music genres and uses the popular GTZAN Dataset for training and evaluation. It focuses on achieving a balance between accuracy and efficiency, making it scalable for further development and real-world applications.

## Project Highlights
- Music Genre Prediction: Classifies audio tracks into ten distinct music genres using a fully connected neural network.
- Audio Feature Extraction: Converts raw audio files into features like MFCCs, chroma, and spectral contrast, enabling effective learning.
- Deep Learning Implementation: Leverages a carefully designed convolutional neural network (CNN) architecture for high-accuracy genre classification.
- Mel Spectrograms: Includes functionality for generating and utilizing Mel Spectrograms to visualize and process audio data as image-like inputs for CNNs.
- Scalability: Structured for easy adaptation to other datasets and integration with more advanced neural architectures.
- Data Augmentation: Enhances the dataset by splitting audio files into smaller clips, improving model generalization and robustness.
- Visualization Tools: Includes tools for visualizing Mel Spectrograms and other extracted audio features, aiding model interpretability.
- Model Training and Evaluation: Provides detailed scripts for training and evaluating the model with hyperparameter tuning options.
- Toolset: Built entirely using Python, with TensorFlow, Keras, and Librosa as the primary libraries.

## Dataset
The GTZAN Dataset is a widely used benchmark dataset for music genre classification. It includes:
* 10 Genres: Blues, classical, country, disco, hip hop, jazz, metal, pop, reggae, and rock.
* 1,000 Audio Files: Each track is 30 seconds long, sampled at 22,050 Hz.
Note: The dataset is not included in this repository. Please download it from Kaggle or another source and place it in the data/ directory before running the project.

## Methodology
### 1. Data Preprocessing:
* Loaded audio files and extracted Mel-frequency cepstral coefficients (MFCCs) and other relevant features.
* Normalized feature values for consistency.
### 2. Model Development:
* Designed and implemented a deep neural network architecture tailored for audio analysis.
* Employed techniques such as dropout and batch normalization to enhance model performance.
### 3. Training and Evaluation:
* Split the dataset into training and testing sets.
* Trained the model on the training data and evaluated it on the testing data.

## Requirements
Ensure you have the following installed before running the project:

* Python 3.7 or later
* TensorFlow 2.x
* Keras
* Librosa
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Installation

### 1. Clone the repository:
git clone https://github.com/yourusername/MusicGenreDL---Deep-Learning-based-Audio-Analysis-and-Music-Genre-Prediction-System.git
### 2. Navigate to the project directory:
cd MusicGenreDL---Deep-Learning-based-Audio-Analysis-and-Music-Genre-Prediction-System
### 3. Install the required libraries:
pip install -r requirements.txt
Note: A GPU-enabled system with CUDA support is recommended for faster training.

## Project Structure

MusicGenreDL/

├── data/                        # Directory for the GTZAN dataset

├── models/                      # Directory to save trained models (optional)

├── output/                      # Directory for storing results (e.g., confusion matrices, reports)

├── MusicGenreDL---Deep-Learning-based-Audio-Analysis-and-Music-Genre-Prediction-System.py  # Main script

├── README.md                    # Project documentation (this file)

## How to Run the Project
### 1. Set Up the Environment:
Install the required libraries using requirements.txt.
### 2. Prepare the Dataset:
Download the GTZAN dataset and place it in the data/ directory.
### 3. Run the Main Script:
Use the following command to execute the script:
python MusicGenreDL---Deep-Learning-based-Audio-Analysis-and-Music-Genre-Prediction-System.py
### 4. Visualize Results:
The output directory contains visualizations, confusion matrices, and classification reports for model evaluation.

## Results
The system effectively classifies music genres by learning intricate patterns in audio features using deep learning models.

## Future Work
- Explore alternative datasets to improve the model's generalization.
- Experiment with more complex architectures such as CNNs or RNNs for enhanced performance.
- Implement real-time music genre classification.

## Acknowledgments
- The GTZAN Dataset was sourced from Kaggle.
- This project was inspired by tutorials and resources from the audio processing and deep learning communities.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## 
Feel free to contribute to this project by submitting issues or pull requests!
