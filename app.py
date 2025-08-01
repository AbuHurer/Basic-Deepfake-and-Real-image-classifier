import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Set title
st.title("🔍 Deepfake Image Detector")

# Load the model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(r"C:\Users\saeem\Desktop\Deepfake\deepfake_model_augmented.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ← NEW
])

class_names = ["fake", "real"]  # ← Double-check based on your training


# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).squeeze()[prediction].item()

    # Map to class name
    class_names = ["fake", "real"]
    predicted_class = class_names[prediction]

    # Display result
    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.markdown(f"### 🔢 Confidence: **{confidence * 100:.2f}%**")
