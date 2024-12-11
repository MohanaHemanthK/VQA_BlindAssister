# Visual Question Answering (VQA) for Accessibility

This repository contains the code and resources for building a Visual Question Answering (VQA) system, aimed at assisting visually impaired individuals. The project leverages deep learning models to answer questions about images, integrating both vision and natural language processing capabilities.

## Team Members
- Mohana Hemanth Kundurthi (Team Leader)
- Venkata Sai Sumanth Aketi (Team Member)
- Shiva Kumar Goud Mucharla (Team Member)

## Dataset Overview
Dataset we used is [**VQA 2.0**](https://visualqa.org/index.html).

- **Purpose:** The dataset is designed to train models to answer open-ended questions about images. These questions require an understanding of vision, language, and commonsense knowledge2.

- **Images:** It contains 265,016 images, including images from the COCO dataset and abstract scenes.

- **Questions:** Each image has at least 3 questions on average, with a total of 1,105,904 questions in the dataset.

- **Answers:** For each question, there are 10 ground truth answers and 3 plausible but likely incorrect answers.

- **Automatic Evaluation:** The dataset includes an automatic evaluation metric to assess the performance of models

## Features
- **Exploratory Data Analysis (EDA)**: Understand the dataset and prepare it for model training.
- **Model Training**: Implements a VGG19 and LSTM-based architecture for image and text processing.
- **Inference**: Includes a streamlined pipeline for deploying the trained model, featuring speech recognition for voice-based queries.
- **Model Evaluation**: Provides scripts to validate the model's performance using various metrics.

---

## File Overview
### `EDA&VQA_Training.ipynb`
- Conducts EDA and preprocessing of the dataset.
- Implements the VQA model using VGG19 for image feature extraction and LSTMs for text processing.
- Training pipeline with visualizations of loss and accuracy.

### `InferenceModelV3.ipynb`
- Prepares the trained model for inference.
- Integrates voice input using `SpeechRecognition` and `gtts (Google Text To Speech)` for user-friendly interaction.
- Generates responses to queries about given images.

### `Model_testing.ipynb`
- Tests the trained model on a separate validation dataset.
- Computes evaluation metrics such as accuracy, precision, recall, and F1 score.
- Includes visualizations for result interpretation.

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
2. Install dependencies: (if you running code in local)
   ```
   pip install requirements.txt
   ```
3. Ensure you have the necessary libraries installed:
- TensorFlow
- SpeechRecognition
- OpenCV
- Matplotlib

---

## Usage
### Training the Model
1. Open EDA&VQA_Training.ipynb.
2. Update paths to the dataset and other configurations.
3. Run the notebook to train the model.

### Testing the Model
1. Open Model_testing.ipynb.
2. Configure the path to your test dataset.
3. Run the notebook to view performance metrics and outputs.

### Running the inference
1. Open FinalInferenceModel.ipynb.
2. Load the trained model weights.
3. Provide access to microphone and webcam and then.

---
## Visualizations:

---
## Results:
**Sound ON**

https://github.com/user-attachments/assets/b58ffd22-717c-451d-a60a-687d2e7d0297

