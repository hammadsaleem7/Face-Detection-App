import streamlit as st
import traceback

def run_app():
    try:
        # all your logic goes inside this block
        st.title("MIT Deep Learning - Debiasing Model")

        # Your existing imports and code here
        import torch
        import torch.backends.cudnn as cudnn
        import mitdeeplearning as mdl
        from pathlib import Path

        # GPU check
        if torch.cuda.is_available():
            device = torch.device("cuda")
            cudnn.benchmark = True
            st.success("GPU is available!")
        else:
            st.warning("GPU is not available. Using CPU.")

        # Data
        CACHE_DIR = Path.home() / ".cache" / "mitdeeplearning"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path_to_training_data = CACHE_DIR.joinpath("train_face.h5")

        if path_to_training_data.is_file():
            st.info("Using cached training data")
        else:
            url = "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
            st.info("Downloading training data...")
            torch.hub.download_url_to_file(url, path_to_training_data)
            st.success("Downloaded training data!")

    except Exception as e:
        st.error("An error occurred:")
        st.text(traceback.format_exc())

run_app()
