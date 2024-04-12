# Import the required library
from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import requests
import argparse

# Command-line arguments handling
parser = argparse.ArgumentParser()
parser.add_argument('--url')  # Argument for image URL
parser.add_argument('--image')  # Argument for local image path
args = parser.parse_args()

# Check if URL or local image path is provided
if args.url:
    # Open image from URL and convert to RGB format
    image = Image.open(requests.get(args.url, stream=True).raw).convert('RGB')
elif args.image:
    # Open local image and convert to RGB format
    image = Image.open(args.image).convert('RGB')
else:
    print('Please choose a URL or local image!')
    parser.print_usage()
    exit()

# Load pre-trained BEiT feature extractor
feature_extractor = BeitFeatureExtractor.from_pretrained('yuuki0/test')

# Load pre-trained BEiT model for image classification
model = BeitForImageClassification.from_pretrained('yuuki0/test')

# Extract features from the image
inputs = feature_extractor(images=image, return_tensors="pt")

# Pass the image features through the model
outputs = model(**inputs)

# Get the logits from the model output
logits = outputs.logits

# Calculate probabilities from logits
probs = logits.softmax(-1)

# Get the index of the predicted class
predicted_class_idx = probs.argmax().item()

# Print the predicted class and its probability
print("Predicted class:", model.config.id2label[predicted_class_idx])
print("Probability:", probs[0][predicted_class_idx].item())
