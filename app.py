# streamlit_app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import mitdeeplearning as mdl

st.title("MIT Deep Learning - Debiasing Model")

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    cudnn.benchmark = True
    st.success("GPU is available and ready!")
else:
    st.error("GPU is not available. Please use GPU-supported environment.")

# Download training data
CACHE_DIR = Path.home() / ".cache" / "mitdeeplearning"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
path_to_training_data = CACHE_DIR.joinpath("train_face.h5")

if path_to_training_data.is_file():
    st.info("Using cached training data")
else:
    with st.spinner("Downloading training data..."):
        url = "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
        torch.hub.download_url_to_file(url, path_to_training_data)
        st.success("Download completed.")
