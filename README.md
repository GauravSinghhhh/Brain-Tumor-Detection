# Brain Tumor Detection Using CNN

## About the Project
This project is a Deep Learning-based Brain Tumor Classification system that uses Convolutional Neural Networks (CNN) to classify brain MRI images into two categories:

- **Tumor Detected:** MRI images containing a brain tumor.
- **No Tumor:** MRI images without a brain tumor.

The project also provides an interactive web interface built using Streamlit, where users can upload an MRI image and get a prediction from the trained model.

## Technologies Used
- Python
- TensorFlow and Keras
- Convolutional Neural Network (CNN)
- OpenCV
- NumPy
- Pillow
- Scikit-learn
- Streamlit

## Model Architecture
The CNN model consists of:
- Three convolutional layers with 32 filters each.
- ReLU activation functions.
- Max Pooling layers.
- Flatten layer.
- Fully connected Dense layer with 64 neurons.
- Dropout layer (0.5) to reduce overfitting.
- Sigmoid output layer for binary classification.

## Project Structure
```text
Brain Tumor Detection/
├── app.py
├── mainTrain.py
├── maintest.py
├── testing.py
├── Braintumor10epochs.h5
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/GauravSinghhhh/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

**2. Install the required dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the Streamlit application**

```bash
streamlit run app.py
```

The application will open in your browser at the local URL displayed in the terminal.

## Model Training

To train the model, place the MRI dataset in the directory structure expected by `mainTrain.py`, then execute:

```bash
python mainTrain.py
```

The trained model is saved as `Braintumor10epochs.h5`.

## Dataset
The MRI image dataset is not included in this repository.

## Disclaimer
This project is developed for educational purposes only. Its predictions should not be used as a substitute for professional medical diagnosis.