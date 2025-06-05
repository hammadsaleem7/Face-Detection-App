import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mitdeeplearning as mdl
import os
from pathlib import Path

# --- App Config ---
st.set_page_config(page_title="Debiasing Facial Detection", layout="centered")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Title & Intro ---
st.title("🧠 Debiasing Facial Detection Systems")
st.markdown("""
Explore algorithmic bias in face detection models, adapted from MIT's 6.S191 Deep Learning course.

Key features:
- 📦 Dataset viewer
- 🤖 CNN face classifier
- 🧪 Model training & metrics
- 📸 Real-time inference
""")

# --- Video Explanation ---
st.header("📺 Why Bias in AI Matters")
st.video("https://www.youtube.com/watch?v=59bMh59JQDo")

# --- Dataset Loading ---
st.header("📦 Load Dataset (CelebA + ImageNet subset)")
CACHE_DIR = Path.home() / ".cache" / "mitdeeplearning"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
path_to_training_data = CACHE_DIR / "train_face.h5"

if not path_to_training_data.exists():
    with st.spinner("Downloading dataset..."):
        url = "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
        torch.hub.download_url_to_file(url, str(path_to_training_data))
        st.success("Download complete!")

loader = mdl.lab2.TrainingDatasetLoader(str(path_to_training_data), channels_last=False)
images, labels = loader.get_batch(100)
H, W = images.shape[2], images.shape[3]

# --- Show Dataset Samples ---
st.subheader("🖼️ Sample Images")
face_images = images[labels == 1].permute(0, 2, 3, 1).numpy()
not_face_images = images[labels == 0].permute(0, 2, 3, 1).numpy()

idx_face = st.slider("Face Image Index", 0, min(50, len(face_images)-1), 10)
idx_nonface = st.slider("Non-Face Image Index", 0, min(50, len(not_face_images)-1), 10)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(face_images[idx_face])
ax[0].set_title("Face")
ax[1].imshow(not_face_images[idx_nonface])
ax[1].set_title("Not Face")
for a in ax: a.axis("off")
st.pyplot(fig)

# --- Model Definition ---
st.header("🤖 CNN Face Detector")

n_filters = 12
in_channels = images.shape[1]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))

def make_standard_classifier(n_outputs=1):
    model = nn.Sequential(
        ConvBlock(in_channels, n_filters, 5, 2, 2),
        ConvBlock(n_filters, 2*n_filters, 5, 2, 2),
        ConvBlock(2*n_filters, 4*n_filters, 3, 2, 1),
        ConvBlock(4*n_filters, 6*n_filters, 3, 2, 1),
        nn.Flatten(),
        nn.Linear((H // 16) * (W // 16) * 6 * n_filters, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, n_outputs),
    )
    return model.to(device)

model = make_standard_classifier()

# --- Model Training ---
st.header("🧪 Train the Model")

epochs = st.slider("Epochs", 1, 20, 5)
batch_size = st.slider("Batch Size", 8, 64, 32)
learning_rate = st.number_input("Learning Rate", value=0.001, format="%.5f")
start_training = st.button("Start Training")

if start_training:
    model = make_standard_classifier()  # Reset model
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    X_all, y_all = loader.get_batch(1000)
    X_all, y_all = X_all.to(device), y_all.to(device).float().unsqueeze(1)

    losses = []
    accuracies = []

    for epoch in range(epochs):
        permutation = torch.randperm(X_all.size(0))
        epoch_loss = 0
        correct = 0

        for i in range(0, X_all.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            X_batch, y_batch = X_all[indices], y_all[indices]

            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == y_batch.byte()).sum().item()

        acc = correct / X_all.size(0)
        losses.append(epoch_loss)
        accuracies.append(acc)

        st.write(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {acc:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(losses)
    ax[0].set_title("Training Loss")
    ax[1].plot(accuracies, color='green')
    ax[1].set_title("Training Accuracy")
    st.pyplot(fig)

# --- Inference ---
st.header("📸 Upload Image for Face Detection")
uploaded_file = st.file_uploader("Upload a face or non-face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    model.eval()
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((W, H))
    img_tensor = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    st.image(img_resized, caption="Uploaded Image", use_column_width=True)

    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.sigmoid(output).item()

    st.subheader("📍 Prediction")
    st.write("**Face**" if prediction > 0.5 else "**Not Face**")
    st.progress(prediction)

# --- Footer ---
st.markdown("---")
st.markdown("© 2025 MIT 6.S191 • Streamlit adaptation by [You]")

