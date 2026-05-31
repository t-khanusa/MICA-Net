# 🚀 MICA-Net: A Multimodal Cross-Attention Network for Human Action Recognition

This repository contains code and instructions for using MICA-Net, a new multimodal action recognition framework that combines data from multiple sensors to enhance human action recognition. Deploy MICA-Net model on NVIDIA Jetson AGX Xavier edge device, and test on a human-machine interactive system with home device control application.

<p align="center">
  <img src="https://github.com/drkhanusa/MICA-Net/blob/main/images/Overview.JPG"/>
</p>

## 🔍 1. Overview

The paper **"MICA-Net: A Multimodal Cross-Attention Network for Human Action Recognition"** presents a solution by designing a compact, wrist-worn device similar to a smartwatch, which facilitates easy portability and mobility. Moreover, we propose a new action recognition method called MICA-Net that uses a cross-attention mechanism to simultaneously extract correlations from multimodal data collected by different sensors, thereby improving model efficiency. The paper has three key contributions:

- 🎯 Introduce **MICA-Net, a new framework for efficient multimodal human action recognition. Validated on three benchmarks** demonstrates superior performance compared to single modality methods and some existing combined methods.
- ⌚ **A new wrist-worn sensor prototype is presented**, featuring a more compact, robust, and user-friendly design. This prototype, which can function as a smartwatch, is easily deployable in ubiquitous environments. It is suitable for various applications, including human life logging and human-machine interaction.
- 🏠 **Developing an application for controlling home appliances using hand gestures** collected by our prototype, running on an edge device, demonstrating its potential for practical use.

## 📋 2. Requirements

- 🐍 Python 3.x
- 🔥 PyTorch
- 📦 torchvision
- 🎥 OpenCV
- 🔢 NumPy

## 🛠 3. Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/drkhanusa/MICA-Net.git
   cd MICA-Net
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 4. Usage

### 📂 4.1. Data Preparation

- **📥 Download Datasets**: [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html), [UESTC-MMEA-CL](https://ivipclab.github.io/publication_uestc-mmea-cl/mmea-cl/), and [MuWiGes](https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/MuWiGes.html) datasets.

- **⚙️ Preprocess**: With inertial data, we use the Gramian Angular Different Field algorithm to transform time series data into 8-dimensional images. With image data, for each video, we use 16 frames, resizing to 172x172. All preprocessing algorithms are in the **algorithm.py** file

### 🏋️ 4.2. Training the Model

1. **📝 Configure Training**: Adjust the setting parameters in each training file of each model: GAFormer model, MoViNet model, and MICA-Net model.
2. **▶️ Run training**:

- With GAFormer model:
  ```bash
  python Training_GAFormer.py 
  ```
- With MoViNet model:
  ```bash
  python Training_MoViNet.py
  ```
- After finishing training both GAFormer and MoViNet models, cutting the last fully connected layer, freezing the previous layers and starting inference, the output is obtained in both the two model branches (inertial and visual) as feature vectors, which will be saved as numpy files (.npy) in separate folders. After completing the above work, training the MICA-Net fusion model with the following command:
  ```bash
  python Training_MICA-Net.py
  ```

### 📊 4.3. Validation

We evaluate the results of the models by parameters: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, T-SNE Graph. These results will also be displayed immediately after you finish training the model.

| 📚 Datasets   | 🎯 Accuracy(%) | 🎯 Precision(%) | 🎯 Recall(%) | 🎯 F1-Score(%) |
| ------------- | -------------- | --------------- | ------------ | -------------- |
| UTD-MHAD      | 94.72          | 94.33           | 94.18        | 94.22          |
| UESTC-MMEA-CL | 99.22          | 99.28           | 99.28        | 99.37          |
| MuWiGes       | 98.98          | 99.00           | 98.92        | 99.16          |
| MMAct(Cross-Subject)       | 89.74          | 90.03           | 88.17        | 89.24    |
| MMAct(Cross-Session)       | 96.67          | 97.12           | 95.95        | 96.57    |

## ⚡ 5. Running on NVIDIA Jetson AGX Xavier

### 🖥 5.1. Set up Jetson AGX Xavier:

1. **🛠 Install JetPack SDK**: Follow NVIDIA's instructions to install JetPack SDK using SDK Manager. In this paper, we use Jetson AGX Xavier 32GB RAM with JetPack version 5.1.2
2. **📂 Transfer Code**: Transfer the MICA-Net code to the Jetson AGX Xavier

### 🚀 5.2 Inference on Jetson AGX Xavier

<p align="center">
  <img src="https://github.com/drkhanusa/MICA-Net/blob/main/images/TensorRT.JPG"/>
</p>

1. **🔄 Convert PyTorch model into .ONNX model**:
   ```bash
   python convertPytorch2onnx.py
   ```
2. **⚡ Optimize Model**: TensorRT is a neural network model optimization library. On NVIDIA's family of edge devices, the **trtexec** tool (TensorRT library) is available to convert ONNX models to TensorRT. Convert into the TensorRT model with the following command:
   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx=MoViNet.onnx --saveEngine=MoViNet.trt --fp16
   ```
3. **▶️ Run inference**: Inference and evaluate results of TensorRT models on Jetson AGX Xavier for speed and accuracy:
   ```bash
   python TensorRT_validation.py
   ```


