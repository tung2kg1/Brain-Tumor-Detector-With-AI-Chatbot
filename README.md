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
