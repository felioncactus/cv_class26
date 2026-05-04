from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "artifacts" / "best_model.pth"
IMAGE_SIZE = 224


def image_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_model(architecture, num_classes):
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unknown architecture: {architecture}")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    config = checkpoint["config"]
    model = build_model(config["architecture"], len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, config, device


@torch.inference_mode()
def predict(image, model, class_names, device, top_k=5):
    tensor = image_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(tensor), dim=1)[0]
    scores, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))
    return [(class_names[i.item()], scores[n].item()) for n, i in enumerate(indices)]


def main():
    st.set_page_config(page_title="Pokemonika", page_icon="P", layout="centered")
    st.title("Pokemonika")
    st.caption("Pokemon classifier demo")

    if not MODEL_PATH.exists():
        st.warning("No trained model found. Run all cells in info.ipynb first.")
        st.stop()

    model, class_names, config, device = load_model()

    uploaded = st.file_uploader(
        "Drop or upload a Pokemon image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input image", use_container_width=True)
        predictions = predict(image, model, class_names, device)
        best_name, best_score = predictions[0]
        st.subheader(best_name)
        st.metric("Confidence", f"{best_score * 100:.2f}%")
        st.write("Top predictions")
        for name, score in predictions:
            st.progress(score, text=f"{name}: {score * 100:.2f}%")
        with st.expander("Model"):
            st.json(config)
    else:
        st.info("Upload an image to classify a Pokemon.")


def is_running_with_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False

    return get_script_run_ctx() is not None


if is_running_with_streamlit():
    main()
elif __name__ == "__main__":
    print("Run this app with: streamlit run c:/cv_class/Pokemonika/gui.py")
