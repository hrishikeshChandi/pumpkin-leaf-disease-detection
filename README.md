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
  I have used EfficientNet-B0 (pre-trained on ImageNet) and fine-tuned it for pumpkin leaf images.

- **Visual Feedback:**

After training, a confusion matrix helps visualize how well the model performed.

- **Streamlit Web App:**
  Upload an image, get a prediction, and see the confidence score all in one click.

- **Metrics & Logs:**
  Training progress is saved and can be viewed later using TensorBoard.

## Model Performance

- **Test Accuracy:** 87.2%
- **Test Loss:** 0.369

These metrics were obtained after training the EfficientNet-B0 model on the pumpkin leaf disease dataset. I have fine-tuned the model using transfer learning from the ImageNet weights. The accuracy and loss values are based on the final validation results from the training log.

## How It Works

1. The model is trained using transfer learning on the provided pumpkin leaf dataset (see Kaggle link below).
2. After training, you can launch the web app and upload images for instant predictions.
3. The app processes the image and tells you what disease (if any) it detects.

## Dataset

This project uses a public dataset from Kaggle:

**Link:** [https://www.kaggle.com/datasets/rifat963/pumpkin](https://www.kaggle.com/datasets/rifat963/pumpkin)

You do not need to manually download or organize the dataset. The code in `effecientnet_b0.ipynb` will automatically download and set up the dataset for you using KaggleHub.

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

3. **Train the model:**

   - Use the `efficientnet_b0.ipynb` and run all the cells.
   - This will save the trained model to `pumpkin_results/model.pth`.
   - Training logs and graphs go to the `pumpkin_results/` folder.
   - The notebook will output a confusion matrix image and save accuracy/loss logs, but does not generate a classification report file.
   - To view logs in TensorBoard:

     ```bash
     tensorboard --logdir pumpkin_results
     ```

4. **Launch the web app:**

   ```bash
   streamlit run app.py
   ```

   - Streamlit will show a local URL in the terminal.
   - Open it in your browser, upload a pumpkin leaf image, and get the prediction instantly.

## Project Structure

- `efficientnet_b0.ipynb` – Jupyter notebook for model training and evaluation.
- `app.py` – The Streamlit web app for predictions.
- `pumpkin_results/` – Stores the trained model, logs, and confusion matrix.
- `requirements.txt` – Lists all required Python packages.

## Notes

- The project uses EfficientNet-B0 for a balance of speed and accuracy.
- Works best with a GPU for training, but CPU is fine for using the web app.
- There is no need to manually download the dataset, the code will handle everything (downloading, extracting and loading)

## License

This project is under the MIT License.
