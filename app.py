import streamlit as st
import IPython
import torch
import mitdeeplearning as mdl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(page_title="Facial Detection Debiasing", layout="wide")

# Title
st.title("Facial Detection Debiasing System")
st.markdown("""
This app demonstrates a facial detection model that learns the latent variables underlying face image datasets 
and uses this to adaptively re-sample the training data, mitigating biases that may be present.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This project explores approaches to address algorithmic bias in facial detection systems.
The model learns latent variables from face image datasets to create debiased models.
""")

# Check for GPU
if not torch.cuda.is_available():
    st.error("GPU is not available. Please ensure you're running this with GPU support.")
    st.stop()

# Main content
def main():
    st.header("Introduction to Algorithmic Bias")
    
    # Embed YouTube video
    st.subheader("Understanding Algorithmic Bias")
    st.video("https://www.youtube.com/watch?v=59bMh59JQDo")
    
    # Dataset section
    st.header("Datasets")
    st.markdown("""
    We use three main datasets:
    1. **Positive training data**: CelebA Dataset (200K+ celebrity faces)
    2. **Negative training data**: ImageNet (non-human categories)
    3. **Test dataset**: Balanced for skin tone and gender representation
    """)
    
    # Load data
    if st.button("Load Training Data"):
        with st.spinner("Loading data..."):
            CACHE_DIR = Path.home() / ".cache" / "mitdeeplearning"
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            path_to_training_data = CACHE_DIR.joinpath("train_face.h5")
            
            if not path_to_training_data.is_file():
                st.info("Downloading training data...")
                url = "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
                torch.hub.download_url_to_file(url, path_to_training_data)
            
            # Load the dataset
            channels_last = False
            loader = mdl.lab2.TrainingDatasetLoader(path_to_training_data, channels_last=channels_last)
            
            # Get batch of images
            number_of_training_examples = loader.get_train_size()
            (images, labels) = loader.get_batch(100)
            B, C, H, W = images.shape
            
            st.success(f"Data loaded successfully! Number of training examples: {number_of_training_examples}")
            
            # Display sample images
            st.subheader("Sample Images from Dataset")
            
            # Create a grid of images
            cols = st.columns(5)  # 5 columns
            for i in range(10):  # Show first 10 images
                with cols[i % 5]:
                    # Convert tensor to numpy and transpose if needed
                    img = images[i].numpy()
                    if not channels_last:
                        img = img.transpose(1, 2, 0)
                    
                    # Normalize if needed
                    if img.max() > 1:
                        img = img / 255.0
                    
                    st.image(img, caption=f"Image {i+1}", use_column_width=True)
            
            # Show dataset statistics
            st.subheader("Dataset Statistics")
            st.write(f"Batch shape: {images.shape}")
            st.write(f"Number of positive examples: {labels.sum()}")
            st.write(f"Number of negative examples: {len(labels) - labels.sum()}")
    
    # Model section
    st.header("Debiasing Model")
    st.markdown("""
    The facial detection model learns latent variables from the training data to:
    - Identify potential biases in the dataset
    - Adaptively re-sample the training data
    - Mitigate biases during training
    """)
    
    # Placeholder for model training and evaluation
    if st.button("Train Model (Placeholder)"):
        st.warning("Model training functionality would be implemented here in a full application.")
        st.write("This would involve:")
        st.write("- Defining the model architecture")
        st.write("- Setting up the training loop")
        st.write("- Implementing the debiasing approach")
        st.write("- Evaluating model performance on balanced test data")
    
    # Results section
    st.header("Expected Results")
    st.markdown("""
    A properly debiased model should show:
    - Similar accuracy across different demographic groups
    - Reduced disparity in false positive/negative rates
    - More equitable performance overall
    """)
    
    # Placeholder for results visualization
    if st.checkbox("Show Sample Results (Placeholder)"):
        st.write("Sample bias metrics across different groups:")
        
        # Create sample data
        groups = ["Lighter Skin", "Darker Skin", "Female", "Male"]
        accuracy = [0.92, 0.89, 0.91, 0.90]
        false_positives = [0.05, 0.08, 0.06, 0.07]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.bar(groups, accuracy)
        ax1.set_title("Accuracy by Group")
        ax1.set_ylim(0.8, 1.0)
        
        ax2.bar(groups, false_positives)
        ax2.set_title("False Positive Rate by Group")
        ax2.set_ylim(0, 0.1)
        
        st.pyplot(fig)
        
        st.write("Note: These are placeholder values for demonstration purposes.")

if __name__ == "__main__":
    main()
