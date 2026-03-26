import numpy as np
from PIL import Image, ImageDraw
from prediction.predict import DigitPredictor
import matplotlib.pyplot as plt

# Create a test image of a digit '5'
img = Image.new('L', (280, 280), 'white')
draw = ImageDraw.Draw(img)

# Draw a big '5'
draw.line([(100, 100), (100, 200)], fill='black', width=20)
draw.line([(100, 100), (180, 100)], fill='black', width=20)
draw.line([(180, 100), (180, 150)], fill='black', width=20)
draw.line([(180, 150), (100, 150)], fill='black', width=20)
draw.line([(100, 150), (100, 200)], fill='black', width=20)
draw.line([(100, 200), (180, 200)], fill='black', width=20)

# Save and display
img.save('test_digit.png')
plt.imshow(img, cmap='gray')
plt.title("Test Digit '5'")
plt.show()

# Predict
predictor = DigitPredictor()
img_resized = img.resize((28, 28))
img_array = np.array(img_resized)
print(f"Original image array - min: {img_array.min()}, max: {img_array.max()}")
print(f"Mean pixel value: {img_array.mean()}")

# Invert if needed
if np.mean(img_array) > 127:
    print("Inverting colors...")
    img_array = 255 - img_array

img_array = img_array.astype('float32') / 255.0
print(f"After preprocessing - min: {img_array.min()}, max: {img_array.max()}")

predictions, digit = predictor.predict_from_array(img_array)
print(f"Predictions: {predictions}")
print(f"Predicted digit: {digit}")

# Show preprocessed image
plt.figure()
plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title("Preprocessed Image Sent to Model")
plt.show()
