Overview
This project implements a handwriting recognition system for the Devanagari script, primarily used in languages such as Hindi, Marathi, and Nepali. Utilizing deep learning and computer vision techniques, the application captures handwritten input in real-time and accurately recognizes the drawn characters.

Features
Real-Time Recognition: Draw Devanagari characters and receive instant feedback on recognition.
Convolutional Neural Network: Trained model with high accuracy based on a comprehensive dataset of handwritten characters.
Interactive Interface: User-friendly drawing area that mimics a blackboard, allowing for easy character input.
Color Detection: Isolates drawn characters using specific color thresholds for effective recognition.
Technology Stack
Deep Learning: Keras (with TensorFlow backend)
Computer Vision: OpenCV
Data Manipulation: Pandas, NumPy

Dataset
The dataset used for training the model consists of Devanagari characters. You can download it using the following link:
https://www.kaggle.com/datasets/rishianand/devanagari-character-set

Workflow
Model Training: Trains a CNN on a dataset of grayscale images of Devanagari characters.
Video Capture: Uses webcam input to detect and process drawn characters.
Character Prediction: Processes images and predicts the character using the trained model.
Results
The system demonstrates effective recognition of handwritten Devanagari characters, showcasing the potential of AI in language processing.
