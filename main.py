import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Handwritten Digit Recognizer')
    parser.add_argument('--mode', type=str, default='gui',
                       choices=['train', 'gui', 'predict'],
                       help='Mode to run the application')
    parser.add_argument('--image', type=str, help='Image path for prediction')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training mode...")
        print("This will train the model on MNIST dataset")
        from training.train import Trainer
        trainer = Trainer()
        history = trainer.train_model()
        
    elif args.mode == 'gui':
        print("Starting GUI application...")
        from prediction.gui_app import DigitRecognizerGUI
        app = DigitRecognizerGUI()
        app.run()
        
    elif args.mode == 'predict':
        if not args.image:
            print("Please provide an image path with --image")
            sys.exit(1)
            
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found!")
            sys.exit(1)
            
        print(f"Predicting digit from image: {args.image}")
        from prediction.predict import DigitPredictor
        predictor = DigitPredictor()
        predictions, digit = predictor.predict_from_image(args.image)
        print(f"Predicted digit: {digit}")
        print(f"Confidence: {predictions[digit]*100:.2f}%")
        
if __name__ == '__main__':
    main()
