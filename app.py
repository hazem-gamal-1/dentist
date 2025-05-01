from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Import the model and treatments
from model import DentalModel, Config
from treatments import treatments

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model.pth"  # Update with your model path

# Initialize config
config = Config()

# Initialize model
model = DentalModel(num_classes=config.NUM_CLASSES, model_name="efficientnet_b2")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define transforms for inference
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    """Make prediction on a single image"""
    # Apply transforms
    image = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    class_name = config.CLASS_NAMES[predicted_class]
    
    # Create a dict of all class probabilities
    all_probs = {config.CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    # Generate chart
    plt.figure(figsize=(10, 6))
    names = list(all_probs.keys())
    values = list(all_probs.values())

    # Sort by probability
    sorted_indices = np.argsort(values)[::-1]
    names = [names[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    # Plot horizontal bar chart
    bars = plt.barh(names, values, color='#4A7AFF')

    # Add percentage labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{values[i]:.2%}', va='center')

    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    plt.tight_layout()
    
    # Save plot to memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    chart_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Get treatment
    treatment_key = class_name.lower()
    # Convert keys to match the treatment dict
    treatment_key = treatment_key.replace("dental ", "")
    if treatment_key == "mouth ulcer":
        treatment_key = "mouth ulcers"
    
    treatment = treatments.get(treatment_key, "No specific treatment found for this condition.")
    
    return class_name, confidence, chart_img, treatment

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Save the uploaded image
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        file.save(img_path)
        
        # Open and process the image
        image = Image.open(img_path).convert('RGB')
        
        # Get prediction
        class_name, confidence, chart_img, treatment = predict(image)
        
        return jsonify({
            'class': class_name,
            'confidence': f"{confidence:.2%}",
            'chart': chart_img,
            'treatment': treatment,
            'image_path': img_path
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
