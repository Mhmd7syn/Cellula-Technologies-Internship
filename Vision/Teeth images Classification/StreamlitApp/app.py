import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="Teeth Image Classification",
    page_icon="🦷",
    layout="centered"
)

# Title and description
st.title("🦷 Teeth Image Classification")
st.markdown("""
Upload an image of teeth, and the custom CNN model will classify it into one of the 7 categories:
**CaS, CoS, Gum, MC, OC, OLP, OT**
""")

# Model Configuration
IMG_SIZE = 224
NUM_CLASSES = 7
CLASSES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
MODEL_PATH = "../Models/CNN_best_model.pth"

# 1. Define the Model Architecture
class TeethClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(TeethClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), # inplace = true: save memory as The computer modifies the data directly in the original memory location.
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 6
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 2. Load the Model
@st.cache_resource
def load_model():
    print("Loading model...")
    model = TeethClassifier(num_classes=NUM_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Check if the path exists
    if not os.path.exists(MODEL_PATH):
       print(f"Model path does not exist: {MODEL_PATH}")
       st.error(f"Error: Model weights not found at `{MODEL_PATH}`. Please double check the path.")
       return None

    # Load weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("Model loaded successfully")
        return model, device
    except Exception as e:
        print(f"Exception loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model_data = load_model()

# 3. Define Preprocessing Transforms
# We use the 'test/val' transforms from the notebook
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Predict Function
def predict(image, model, device):
    # Preprocess the image
    img_tensor = data_transforms(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        # Convert LogSoftmax output to probabilities using exp()
        probabilities = torch.exp(output)[0]
        
    return probabilities.cpu().numpy()

# 5. UI functionality
if model_data is not None:
    model, device = model_data

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Add a submit button
        if st.button("Classify Image"):
            with st.spinner("Analyzing..."):
                # Run prediction
                probs = predict(image, model, device)
                
                # Get the top prediction
                predicted_class_idx = np.argmax(probs)
                predicted_class = CLASSES[predicted_class_idx]
                confidence = probs[predicted_class_idx] * 100
                
                # Display Result
                st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")
                
                # Display probabilities as a bar chart
                st.subheader("Class Probabilities")
                
                # Create a DataFrame for the bar chart
                prob_df = pd.DataFrame({
                    'Class': CLASSES,
                    'Probability': probs * 100
                })
                # Set class as index for st.bar_chart
                prob_df.set_index('Class', inplace=True)
                
                st.bar_chart(prob_df)
else:
    st.warning("Model could not be loaded. Inference is disabled.")
