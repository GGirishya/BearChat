# Missouri State University (MSU) Chatbot - Fine-Tuned Llama 3.1 8B

A domain-specific AI assistant for Missouri State University. This project fine-tunes the Llama-3.1-8B-Instruct model on verified MSU data and the development server is built via a Flask API with support for RAG (Retrieval-Augmented Generation), PDF/Image document analysis, and live web search.

## Project Structure

```
├── finetune_8b_model.py          # Main fine-tuning script (LoRA + Nvidia A100 optimized)
├── api_server.py                 # Flask API server (Chat, Doc Upload, Web Search)
├── web_scrapping_script.py       # Production-grade data scraper (Trafilatura + Playwright)
├── web_search.py                 # Google Custom Search integration module
├── requirements.txt              # Dependencies
├── Json_data_storage/            # Directory for training datasets
    |-- Json_data_storage/new_training_data_msu.json
    |-- Json_data_storage/hallucination_fixing_patches_data.json
└── models/                       # Directory where adapters are saved
    └── latest/                   # The verified fine-tuned model adapter
```

---

## Setup Guide ("Compilation")

### 1. Prerequisites
- **Python 3.10+**
- **NVIDIA GPU** (A100 recommended for training, T4/L4 for inference) or Apple Silicon (Mac M1/M2/M3/M4).
- **Hugging Face Account** with access to [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

### 2. Installation
Clone the repository and install dependencies:

```bash
# Clone
git clone https://github.com/GGirishya/BearChat.git
cd Fine-tunned-project-v3.0

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python Dependencies
pip install -r requirements.txt
```

### 3. Install Third-Party Tools
This project uses specialized tools that require system-level installation:

**A. Playwright (for Web Scraping)**
Required for `web_scrapping_script.py` to handle JavaScript-heavy sites.
```bash
pip install playwright
playwright install chromium
```

**B. Tesseract OCR (for Image/PDF Analysis)**
Required for `api_server.py` to read scanned documents.
- **Mac**: `brew install tesseract`
- **Windows**: [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki)

### 4. Configuration
Create a `.env` file in the project root:
```env
# Hugging Face Token (Required for Model)
HF_TOKEN=hf_your_token_here

# Google Custom Search (Optional, for live web search)
GOOGLE_API_KEY=your_google_key
GOOGLE_CSE_ID=your_cse_id
```

---

## Usage Guide

### Phase 1: Data Collection & Preparation
Use the scraper to build your training dataset.

**Run the Scraper:**
```bash
python web_scrapping_script.py
```
- **Interactive Mode**: Enter URLs one by one (e.g., `https://missouristate.edu/admissions`).
- **Output**: Generates `msu_training_YYYYMMDD.json` ready for training.

### Phase 2: Fine-Tuning the Model
Train the model on your collected data. The script uses LoRA for efficiency.

**Run Training:**
```bash
python finetune_8b_model.py
```
1.  **Select Data**: The script will ask for the path to your JSON file. Press Enter to use the latest detected file.
2.  **Training**: Runs for ~3 epochs using A100 optimizations, (Flash Attention 2, bfloat16), if sample size ~1000+.
3.  **Output**: Saves the adapter to `checkpoints/latest/`.


### Phase 3: Deployment (API Server)
Host the model for inference.

**Start the Server:**
```bash
python api_server.py
```
The server starts at `http://localhost:8080`.

#### API Endpoints

**1. Chat (`POST /chat`)**
Standard conversation endpoint.
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the CS degree requirements?",
    "max_length": 512
  }'
```

**2. Document Analysis (`POST /upload`)**
Upload a PDF (transcript, syllabus) or Image to ask questions about it.
```bash
curl -X POST http://localhost:8080/upload \
  -F "file=@transcript.pdf" \
  -F "question=What is my GPA?"
```

**3. Web Search Integration**
To enable live web search, ensure `GOOGLE_API_KEY` is set in `.env` and pass `web_search_enabled: true` in the`/chat` payload.

---

## Third-Party Tools Explained

This project leverages powerful open-source tools:

1.  **Hugging Face (Transformers, PEFT, TRL)**:
    -   *Purpose*: Core machine learning framework. `transformers` loads the model, `peft` handles the lightweight LoRA adapters, and `trl` manages the Supervised Fine-Tuning loop.

2.  **PyMuPDF (fitz)**:
    -   *Purpose*: Extremely fast PDF text extraction. Used in `api_server.py` to process uploaded documents 10x faster than standard libraries.

3.  **Trafilatura**:
    -   *Purpose*: The engine behind `web_scrapping_script.py`. It extracts clean content from messy HTML, removing ads and navigation bars automatically.

4.  **Playwright**:
    -   *Purpose*: A headless browser used by the scraper to render JavaScript-heavy pages (like Single Page Applications) before extraction.

5.  **Google Custom Search JSON API**:
    -   *Purpose*: Provides real-time internet access for the chatbot to answer questions about recent events (Academic Calendar changes, News) not in the training data.

---

## Running on Google Colab

If you prefer using Google Colab, follow this initialization sequence:

1.  **Dependencies**: Install `requirements.txt`.
2.  **Authentication & Storage**:
    ```python
    from huggingface_hub import login
    login(new_session=False)
    
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3.  **Execution Order**:
    -   `web_scrapping_script.py` (Data Collection)
    -   `finetune_8b_model.py` (Model Training)
    -   `web_search.py` (Search Module)
    -   `gradio_UI_google_collab_only.py` (UI & Inference)
