# Handwritten Digit Recognizer

A deep learning project that recognizes handwritten digits (0-9) using Artificial Neural Networks (ANN) on the MNIST dataset. Achieves 98.68% test accuracy with an interactive GUI for real-time digit recognition.

---

## Overview

This project implements a complete machine learning pipeline for handwritten digit recognition:

- Dataset: MNIST (70,000 handwritten digits)
- Model: Artificial Neural Network with 4 layers
- Accuracy: 98.68% on test data
- Interface: Tkinter GUI for drawing and predicting digits

---

## Features

- Train ANN model on MNIST dataset with 98.68% accuracy
- Interactive GUI to draw and recognize digits in real-time
- Load external images for prediction (PNG, JPG, JPEG, BMP)
- Real-time confidence bars showing probability for digits 0-9
- Save and load trained models
- Confusion matrix and accuracy graphs

---

## Project Structure

handwritten-digit-recognizer/
├── data/                 # Data loading scripts
├── models/               # ANN architecture
├── training/             # Training pipeline
├── prediction/           # GUI and prediction
├── utils/                # Helper functions
├── saved_models/         # Trained models
├── logs/                 # Training visualizations
├── main.py               # Main entry point
└── requirements.txt      # Dependencies

---

## Installation

1. Clone the repository
   git clone https://github.com/YOUR_USERNAME/handwritten-digit-recognizer.git
   cd handwritten-digit-recognizer

2. Create virtual environment
   Windows: python -m venv venv
   Windows: .\venv\Scripts\activate
   Mac/Linux: python3 -m venv venv
   Mac/Linux: source venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt

---

## Usage

Train the model:
   python main.py --mode train

Launch GUI:
   python main.py --mode gui

Predict single image:
   python main.py --mode predict --image "digit.png"

---

## GUI Instructions

1. Draw a digit in the white canvas using your mouse
2. Click "Predict" to recognize the digit
3. View the predicted digit and confidence bars for 0-9
4. Click "Clear" to draw another digit
5. Click "Load Image" to test external images

Drawing Tips:
- Draw thick, bold lines
- Make digits large and centered
- Draw slowly for best results

---

## Model Architecture

Layer 1: Dense (512 units, ReLU, Dropout 0.3)
Layer 2: BatchNormalization
Layer 3: Dense (256 units, ReLU, Dropout 0.3)
Layer 4: BatchNormalization
Layer 5: Dense (128 units, ReLU, Dropout 0.2)
Layer 6: Dense (10 units, Softmax)

Total Parameters: ~500,000

---

## Performance Results

Test Accuracy: 98.68%
Validation Accuracy: 98.97%
Training Accuracy: 99.45%

Digit-wise Accuracy:
Digit: 0   1   2   3   4   5   6   7   8   9
Acc:   99% 99% 98% 98% 97% 98% 99% 98% 98% 97%

Most Confused Pairs: 4↔9, 7↔1, 3↔8

---

## Technologies Used

- Python 3.11 - Programming language
- TensorFlow 2.15 - Deep learning framework
- NumPy - Numerical computations
- Matplotlib - Data visualization
- scikit-learn - Evaluation metrics
- Pillow - Image processing
- Tkinter - GUI framework

---

## Dependencies

tensorflow==2.15.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
opencv-python==4.8.0.74
pillow==10.0.0

---

## Troubleshooting

Problem: ModuleNotFoundError
Solution: Activate virtual environment first

Problem: GUI only predicts '0'
Solution: Draw thicker, larger, centered digits

Problem: Training takes too long
Solution: Reduce epochs in training/train.py

Problem: Model not found
Solution: Train model first with python main.py --mode train

---

## Future Improvements

- Implement Convolutional Neural Network (CNN) for higher accuracy
- Add support for multi-digit recognition
- Deploy as web application using Flask
- Add data augmentation for better generalization
- Create mobile app version

---

## License

MIT License

---

## Author

Your Name
GitHub: @yourusername

---

## Acknowledgments

- MNIST Dataset by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- TensorFlow and Keras teams for the deep learning framework

---

Star this repository if you found it helpful!
