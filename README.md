## Model Deep Dive: Xception Architecture

This project leverages the **Xception** (Extreme Inception) network, a powerful deep learning architecture that pushes the boundaries of depthwise separable convolutions.

### Model Architecture Details
Instead of standard Inception modules, the model is built on **Depthwise Separable Convolutions**, which effectively decouple the mapping of cross-channel correlations and spatial correlations.

*   **Feature Extractor:** The model utilizes a pre-trained **Xception** base (trained on ImageNet). The bottom layers are frozen to retain general feature detection capabilities (edges, textures), while the top layers are fine-tuned to the specific nuances of MRI medical imaging.
*   **Custom Classification Head:**
    *   **Global Average Pooling:** Reduces the spatial dimensions of the feature maps.
    *   **Fully Connected (Dense) Layers:** Multiple layers with **ReLU** activation to learn complex patterns within the brain scans.
    *   **Dropout Regularization:** Set at 0.5 to prevent overfitting by randomly deactivating neurons during training.
    *   **Softmax Output:** A final layer with neurons corresponding to each class (e.g., Glioma, Meningioma, No Tumor, Pituitary).

### Training Specifications
*   **Input Shape:** $299 \times 299 \times 3$ (RGB-scaled MRI slices).
*   **Transfer Learning Strategy:** Fine-tuning. The pre-trained weights provide a robust starting point, significantly reducing training time and data requirements.
*   **Image Augmentation:** To improve generalization, the training pipeline includes random rotations, horizontal flips, and zoom adjustments to simulate variations in patient positioning during MRI scans.
*   **Normalization:** Pixel values are scaled to a range of $[-1, 1]$ to align with the Xception preprocessing requirements.

### Why Xception?
Xception was chosen for this project because it offers a superior balance between **computational efficiency** and **classification accuracy**. By using depthwise separable convolutions, the model has fewer parameters than standard convolutions, making it faster to run on weak hardware while maintaining high precision in identifying subtle tumor boundaries.

## Installation

### Prerequisites
*   Python 3.9+
*   GPU with CUDA support (recommended for inference)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tung2kg1/Brain-Tumor-Detector-With-AI-Chatbot.git](https://github.com/tung2kg1/Brain-Tumor-Detector-With-AI-Chatbot.git)
    cd Brain-Tumor-Detector-With-AI-Chatbot
    ```
    
2.  **Download the model:**

    https://drive.google.com/file/d/1teCpCkbvNYRhxftOEl2X3xgLu_gs8jah/view?usp=drive_link

4.  **Create a virtual environment:**
    ```bash
        python -m venv venv
        source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Environment Variables:**
    To enable the AI chatbot functionality, you must provide a **Google Gemini API Key**. Create a `.env` file in the root directory and add your key as follows:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```
    *Note: This key is used to power the conversational assistant that explains classification results and answers medical-related queries.*

## Usage

Run the following command to launch the web interface:
```bash
streamlit run app.py
```

## Disclaimer

This application is for educational and research purposes only. It is not a certified medical device and should not be used for clinical diagnosis. Always consult with a qualified medical professional for health-related concerns.
