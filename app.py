import torch
import streamlit as st
import torchvision
from torchvision import transforms
from PIL import Image

model = torch.load("./old_results/model.pth", weights_only=False)

st.write("Please upload an image of the leaf which you want to check for diseases...")

upload_image = st.file_uploader(
    label="Upload", accept_multiple_files=False, type=["jpg", "png", "jpeg"]
)

class_names = [
    "Bacterial Leaf Spot",
    "Downy Mildew",
    "Healthy Leaf",
    "Mosaic Disease",
    "Powdery_Mildew",
]


weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

auto_transforms = weights.transforms()

if upload_image:
    img = Image.open(upload_image).convert("RGB")

    # transformed_img = auto_transforms(img).unsqueeze(dim=0)

    img_trans = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    
    transformed_img = img_trans(img).unsqueeze(dim=0)

    model.eval()
    with torch.inference_mode():
        logits = model(transformed_img)
        probs = torch.softmax(logits, dim=1)
        max_prob = probs.max()
        label = class_names[logits.argmax(dim=1).item()]

        st.image(
            img,
            caption=f"Prediction: {label} | Confidence: {max_prob * 100:.2f}%",
            width=300,
        )
