# 🧠 Deepfake Image Detection with PyTorch and ResNet18

Build a **deepfake image classifier** using PyTorch and a pretrained ResNet18 backbone. This project distinguishes between real and fake images, leveraging data augmentation and class imbalance handling for robust results.

## 📂 Dataset

We utilize the publicly available deepfake image dataset from Kaggle.

**Folder structure after extraction:**
```
train/
├── fake/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── real/
    ├── image1.png
    ├── image2.png
    └── ...
```

## 🧾 Project Structure

```
deepfake-image-detection/
├── model.py           # Training script
├── app.py             # Streamlit web app
├── deepfake_model.pth # Saved PyTorch model
├── requirements.txt   # Dependencies
├── README.md          # Project documentation
```

## 🛠 Features

- 🔍 **Image classification:** Real vs Fake detection  
- 🧠 **ResNet18:** Uses pretrained ImageNet weights  
- 🧪 **Data augmentation:** Improves model generalization  
- ⚖️ **Class imbalance:** Weighted sampling for balanced training  
- 💾 **Model checkpointing:** Save models for later use  
- 🌐 **Streamlit interface:** Easily test images through a web UI  

## 🚀 How to Run (Locally)

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/deepfake-image-detection.git
    cd deepfake-image-detection
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Train the model:**
    ```
    python model.py
    ```

4. **Run the Streamlit app:**
    ```
    streamlit run app.py
    ```

## 🧠 Model Details

- **Architecture:** `ResNet18` (`torchvision.models.resnet18`)
- **Input Size:** 224×224
- **Output Classes:** 2 (`real`, `fake`)
- **Loss Function:** CrossEntropy (with class weighting)
- **Optimizer:** Adam
- **Learning Rate:** 0.0005
- **Epochs:** 10
- **Batch Size:** 32  
- **Transforms:**
    - **Training:** Resize, ColorJitter, RandomCrop, Normalize
    - **Validation:** Resize, Normalize

## 📈 Performance

- **Training Accuracy:** ~99%
- **Validation Accuracy:** ~97%

## 🧪 Example Predictions

- Both real and fake images are correctly classified by the model.  
- The Streamlit app provides instant predictions with an image preview.

## 🧾 requirements.txt

```
torch
torchvision
numpy
pillow
streamlit
```

## 📌 Future Improvements

- 🔄 Model quantization or ONNX export for deployment  
- 🧠 Adoption of more complex models, e.g., EfficientNet, ViT  
- 📊 Add confusion matrix and metrics report in the app  
- 🌐 Deploy Streamlit app online (e.g., Streamlit Cloud)

## 📜 License

MIT License

## Contact

Created by **Mohammed Abu Hurer**  
For questions or suggestions, reach out via mohammedabuhurer@gmail.com or open an issue on the repository.
