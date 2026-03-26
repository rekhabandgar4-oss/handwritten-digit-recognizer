import numpy as np
from tensorflow.keras.datasets import mnist
from models.ann_model import HandwrittenDigitANN
import matplotlib.pyplot as plt

print("Loading MNIST data...")
(_, _), (x_test, y_test) = mnist.load_data()

# Load model once outside the loop
print("Loading model...")
model = HandwrittenDigitANN()
model.load_model('saved_models/best_model.h5')

# Test with first 5 images
for i in range(5):
    test_image = x_test[i]
    true_label = y_test[i]
    
    print(f"\nTest {i+1}: True label = {true_label}")
    
    # Preprocess
    test_image = test_image.astype('float32') / 255.0
    test_image = test_image.reshape(1, 784)
    
    # Predict
    prediction_result = model.predict(test_image)
    
    # Check the format and extract prediction
    print(f"Prediction type: {type(prediction_result)}")
    print(f"Prediction shape: {np.array(prediction_result).shape}")
    
    # Handle different possible return formats
    if isinstance(prediction_result, tuple) or isinstance(prediction_result, list):
        predictions = prediction_result[0]
    else:
        predictions = prediction_result
    
    # Ensure predictions is a numpy array
    predictions = np.array(predictions)
    
    # Get the predicted digit
    if predictions.ndim == 1:
        predicted_digit = np.argmax(predictions)
        confidence = predictions[predicted_digit]
    else:
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit]
    
    print(f"Predicted: {predicted_digit} (confidence: {confidence:.4f})")
    print(f"All predictions: {predictions}")
    
    # Show the image
    plt.figure(figsize=(2, 2))
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"True: {true_label}, Pred: {predicted_digit}")
    plt.show()
    
    # Stop after first test to see output
    if i == 0:
        break

print("\n✅ Test completed!")
