import os
import io
import base64
import numpy as np
import torch
import cv2
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, render_template, jsonify, send_file
import segmentation_models_pytorch as smp

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Keep models in a parent folder relative to Flask_App if they exist there
MODEL_PATH = os.path.join(BASE_DIR, '..', 'Models', 'best_Pretrained_unet_model.pth')

# Define our prediction device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Initialize model
print("Loading model...")
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None, # Weights are already in the saved state
    in_channels=4,
    classes=1,
    activation='sigmoid'
)

# 2. Load weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Validation transform matching the training definition
image_size = 224
val_transform = A.Compose([
    A.Resize(image_size, image_size),
    ToTensorV2()
])

def encode_image(image_array, is_bgr=False):
    if is_bgr:
        vis = image_array
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
        vis = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        vis = image_array
    is_success, buffer = cv2.imencode(".png", vis)
    if is_success:
        return base64.b64encode(buffer).decode('utf-8')
    return ""

def process_image(file_stream):
    """Read the bytes of a TIFF file, extract bands, and apply transforms"""
    file_bytes = file_stream.read()
    image = tifffile.imread(io.BytesIO(file_bytes)).astype(np.float32)

    band_indices = [5, 7, 10, 11]
    if image.shape[-1] < max(band_indices):
        raise ValueError(f"Uploaded image does not have enough bands. Expected at least {max(band_indices) + 1}.")

    tensor_input = image[:, :, band_indices]
    augmented = val_transform(image=tensor_input)
    tensor_image = augmented['image'].unsqueeze(0)
    
    # Extract RGB for visualization
    rgb = image[:, :, [3, 2, 1]]
    rgb = np.clip(rgb / 3000.0, 0, 1)
    rgb_255 = (rgb * 255).astype(np.uint8)
    rgb_255 = cv2.resize(rgb_255, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    return tensor_image, rgb_255

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not file.filename.lower().endswith(('.tif', '.tiff')):
         return jsonify({'error': 'Please upload a valid .tif or .tiff file'}), 400

    try:
        # Preprocess
        tensor_image, rgb_255 = process_image(file)
        tensor_image = tensor_image.to(device)

        with torch.no_grad():
            output = model(tensor_image)
            
        pred_mask = (output > 0.5).float().squeeze().cpu().numpy()
        mask_image_255 = (pred_mask * 255).astype(np.uint8)

        bgra_mask = cv2.cvtColor(mask_image_255, cv2.COLOR_GRAY2BGRA)
        bgra_mask[mask_image_255 == 255] = [246, 130, 59, 255] # BGR for Cyan/Blue
        bgra_mask[mask_image_255 == 0] = [0, 0, 0, 255] # Black

        rgb_b64 = encode_image(rgb_255, is_bgr=False)
        pred_b64 = encode_image(bgra_mask, is_bgr=True)
        
        gt_b64 = None
        base_name = os.path.splitext(file.filename)[0]
        gt_path = os.path.join(BASE_DIR, '..', 'Data', 'labels', f"{base_name}.png")
        if os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_mask = cv2.resize(gt_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
                gt_mask = (gt_mask > 0).astype(np.uint8) * 255
                bgra_gt = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGRA)
                bgra_gt[gt_mask == 255] = [246, 130, 59, 255] # match pred mask color
                bgra_gt[gt_mask == 0] = [0, 0, 0, 255]
                gt_b64 = encode_image(bgra_gt, is_bgr=True)

        return jsonify({
            'rgb_image': f"data:image/png;base64,{rgb_b64}",
            'pred_mask': f"data:image/png;base64,{pred_b64}",
            'gt_mask': f"data:image/png;base64,{gt_b64}" if gt_b64 else None
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
