# Pumpkin Leaf Disease Detection

## Overview

This project helps identify diseases in pumpkin leaves using deep learning. You can upload a photo of a pumpkin leaf and get an instant prediction about its condition. Everything runs through a simple web app, powered by a trained image classification model.

## Features

- **Leaf Disease Prediction:**
  The model can classify a leaf image into one of five categories:

  - Bacterial Leaf Spot
  - Downy Mildew
  - Healthy Leaf
  - Mosaic Disease
  - Powdery Mildew

- **Transfer Learning:**
  We’ve used EfficientNet-B0 (pre-trained on ImageNet) and fine-tuned it for pumpkin leaf images.

- **Visual Feedback:**
  After training, a confusion matrix helps visualize how well the model performed.

- **Streamlit Web App:**
  Upload an image, get a prediction, and see the confidence score all in one click.

- **Metrics & Logs:**
  Training progress is saved and can be viewed later using TensorBoard.

## How It Works

1. Provide pumpkin leaf images for training (dataset included below).
2. The model is trained using transfer learning.
3. Once trained, you can launch the web app and upload images for instant predictions.
4. The app processes the image and tells you what disease (if any) it detects.

## Dataset

We used a public dataset from Kaggle:

**Link:** [https://www.kaggle.com/datasets/rifat963/pumpkin](https://www.kaggle.com/datasets/rifat963/pumpkin)

The dataset includes two folders:

- `Original Dataset/` – Raw training images grouped by class.
- `Augmented/` – Additional validation and test images.

Both of these should be placed inside a folder called `Pumpkin/` like this:

```
Pumpkin/
├── Original Dataset/
└── Augmented/
    ├── valid/
    └── test/
```

## Technologies Used

- **PyTorch** – For training the model.
- **TorchVision** – To load EfficientNet and handle image transforms.
- **Streamlit** – To create a lightweight web interface.
- **TensorBoard** – For monitoring training progress.
- **Other libraries:** Pillow, tqdm, mlxtend, torchmetrics.

## Setup and Usage

1. **Clone the repo:**

   ```bash
   git clone https://github.com/hrishikeshChandi/pumpkin-leaf-disease-detection.git
   cd pumpkin-leaf-disease-detection
   ```

2. **Install dependencies:**

   Make sure Python 3.7 or higher is installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the dataset:**

   - Download the dataset from Kaggle:
     [https://www.kaggle.com/datasets/rifat963/pumpkin](https://www.kaggle.com/datasets/rifat963/pumpkin)
   - Extract the contents and place both folders (`Original Dataset/` and `Augmented/`) inside a directory called `Pumpkin/`.

4. **Train the model:**

   - Use the `effecientnet_b0.ipynb` and run all the cells.
   - This will save the trained model to `old_results/model.pth`.
   - Training logs and graphs go to the `old_results/` folder.
   - To view logs in TensorBoard:

     ```bash
     tensorboard --logdir old_results
     ```

5. **Launch the web app:**

   ```bash
   streamlit run app.py
   ```

   - Streamlit will show a local URL in the terminal.
   - Open it in your browser, upload a pumpkin leaf image, and get the prediction instantly.

## Project Structure

- `effecientnet_b0.ipynb` – Contains the code used for training the model.
- `app.py` – The Streamlit web app for predictions.
- `Pumpkin/` – Contains the dataset (as described above).
- `old_results/` – Stores the trained model and logs.
- `requirements.txt` – Lists all required Python packages.

## Notes

- The project uses EfficientNet-B0 for a balance of speed and accuracy.
- Works best with a GPU for training, but CPU is fine for using the web app.
- The dataset setup must follow the directory structure shown above.

## License

This project is under the MIT License.
