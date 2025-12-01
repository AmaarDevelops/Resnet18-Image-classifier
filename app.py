import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import request,jsonify,render_template,url_for,Flask
from PIL import Image
import torchvision
from torchvision import models,transforms
import io
import os


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load model and mapping ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,2)

    state_dict = torch.load('dogcat_resnet18.pth',map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


Model = load_model()

class_idx = torch.load('class_to_index.pth')

# Mapping item classes
idx_to_class = {v : k for k,v in class_idx.items()}

# -- Same transform from Neural network ---

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])


def predict_image(image_bytes):
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Converting into tensor
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = Model(image_tensor)

        probabilties = F.softmax(output,dim=1)[0]

        _,predicted = torch.max(output.data,1)

        predicted_class_name = idx_to_class[predicted.item()]
        confidence = probabilties[predicted.item()].item() * 100

    return predicted_class_name,confidence


# ------- Flask routes ----------

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
        if 'file' not in request.files:
            return jsonify({'error' : 'No file found'}) , 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error' : 'no file attached'}) , 400

        if file:
            try:
                img_bytes = file.read()

                class_name,confidence = predict_image(img_bytes)

                return jsonify({
                    'class' : class_name,
                    'confidence' : f'{confidence:.2f}%',
                    'status' : 'success'
                })

            except Exception as e:
                return jsonify({'error' : f'Prediction failed : {e}'}) , 500


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER,exist_ok=True)
    app.run(debug=True)


