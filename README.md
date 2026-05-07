# German Traffic Sign Recognition (GTSRB) - CNN Classifier

A deep learning solution for **German Traffic Sign Recognition** using Convolutional Neural Networks (CNN). The model achieves **~92.8% accuracy** on the official test set.

![Accuracy Plot](assets/accuracy_plot.png)
![Loss Plot](assets/loss_plot.png)

## 📋 Project Overview

This project implements a **Convolutional Neural Network** to classify 43 different types of German traffic signs. It was built using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset, one of the most popular benchmarks in computer vision for traffic sign classification.

The model is trained from scratch using TensorFlow/Keras and demonstrates strong performance with relatively simple architecture, making it suitable for real-time applications and further research.

### Key Features

- **43-class classification** of German traffic signs
- Custom CNN architecture with dropout regularization
- Data preprocessing and augmentation readiness
- Training visualization (accuracy & loss curves)
- Evaluation on official test set
- Reproducible pipeline (Colab + Python script)

## 🏗️ Model Architecture

```python
- Conv2D (5x5) + ReLU → 32 filters
- Conv2D (5x5) + ReLU → 32 filters
- MaxPooling + Dropout(0.25)
- Conv2D (3x3) + ReLU → 64 filters
- Conv2D (3x3) + ReLU → 64 filters
- MaxPooling + Dropout(0.25)
- Flatten
- Dense(256) + ReLU + Dropout(0.5)
- Dense(43) + Softmax
```

## 📊 Results

- **Test Accuracy**: **92.80%**
- Training Accuracy: ~90.86%
- Validation Accuracy: ~97.32% (after 15 epochs)

## 📁 Repository Structure

```bash
├── code11.py                    # Main training & evaluation script
├── colab_train.ipynb            # Google Colab notebook
├── my_model.h5                  # Trained model (optional - add to .gitignore if large)
├── requirements.txt
├── README.md
└── assets/                      # Screenshots and plots
    ├── accuracy_plot.png
    └── loss_plot.png
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow / Keras**
- **NumPy, Pandas, PIL**
- **Matplotlib, Scikit-learn**
- **KaggleHub** (dataset loading)

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/gtsrb-traffic-sign-classifier.git
cd gtsrb-traffic-sign-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run training
```bash
python code11.py
```

Or open `colab_train.ipynb` in Google Colab for easier execution.

## 📦 Requirements

```txt
tensorflow
keras
numpy
pandas
matplotlib
Pillow
scikit-learn
kagglehub
```

## 📈 Dataset

- **Dataset**: [GTSRB - German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Classes**: 43
- **Training Images**: ~39,209
- **Test Images**: 12,630


## 🙏 Acknowledgments

- [GTSRB Dataset](https://benchmark.ini.rub.de/) by the Institute of Neural Information Processing, University of Ulm
- Kaggle community for the dataset mirror

-
