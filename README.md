# Medical Text Summarisation

A deep learning project for automatically summarizing medical reports using fine-tuned T5 models with LoRA (Low-Rank Adaptation) optimization.

## Overview

This project takes medical findings/reports and generates concise clinical summaries using a LoRA-optimized Google Flan-T5 Small model. It includes:
- **Data Processing**: XML to CSV conversion with cleaning
- **Model Training**: LoRA fine-tuning on medical data
- **Inference**: Batch processing for medical text summarization
- **API**: FastAPI endpoint for serving predictions
- **Evaluation**: ROUGE score metrics and visualization
- **UI**: TO display the results from the predictions

## Project Structure

```
Medical Text Summarisation/
├── api/
│   ├── app.py              # FastAPI application
│   ├── client.py           # API client for testing
│   └── src/
│       ├── trainmodel_v2.py
│       ├── evaluate_trainmodel_v2.py
│       ├── config.py
│       └── data_processing.py
├── src/
│   ├── cleaning_data.py
│   ├── data_processing.py
│   └── config.py
├── data/
│   ├── raw/                # XML medical reports
│   └── processed/          # Cleaned CSV files
├── models/                 # Trained models
├── tests/                  # Unit tests
└── README.md
```

## Installation

### Requirements
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- FastAPI & Uvicorn

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/LavenaD/Medical-Test-Summarisation.git
cd Medical-Test-Summarisation
```

2. **Create virtual environment**
```bash
python -m venv summarise
summarise\Scripts\activate  # Windows
source summarise/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Training steps

```bash
python api/src/trainmodel_v2.py
```

This will:
- Load medical data from `data/raw/`
- Clean and preprocess the data
- Train the T5 model with LoRA fine-tuning
- Save the trained model to `models/medical_summarizer_peft/`


### Running Inference

#### Python Script
```bash
python api/inference.py
```
# Single text
This will:
- Print the summary


#### FastAPI Server

1. **Start the API**
```bash
uvicorn api.app:app --reload
```

2. **Call the endpoint**
```bash
python api/client.py
```

Or use curl:
```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"medical_text": "heart size is normal the lungs are clear no pneumothorax"}'
```

### Evaluation

Generate evaluation metrics and visualizations:

```bash
python api/src/evaluate_trainmodel_v2.py
```

This generates:
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Summary length distribution plots
- Comparison with reference summaries

## API Endpoints

### GET `/`
Check API status
```json
{
  "status": "API is running"
}
```

### POST `/summarize`
Generate medical summary

**Request:**
```json
{
  "medical_text": "medical findings text here"
}
```

**Response:**
```json
{
  "input": "medical findings text here",
  "summary": "generated summary"
}
```

## Testing

Run unit tests:
```bash
pytest tests/
```

## Configuration

Edit `src/config.py` to customize:
- Model name and paths
- Data directory paths
- Training hyperparameters
- Input/output text lengths

```python
MODEL_NAME = "google/flan-t5-small"
TEXT_COL = "findings"
TARGET_COL = "impression"
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256
```

## Model Details

- **Base Model**: Google Flan-T5 Small (80M parameters)
- **Fine-tuning**: LoRA with r=8, alpha=16
- **Optimization**: AdamW optimizer
- **Device**: CPU optimized (GPU compatible)

## Data Format

Input CSV format:
```
findings,impression
"Patient with chest pain, CXR findings...","No acute cardiopulmonary abnormality"
```

## Results

evaluation_results.csv stores the actual and predicted impressions for the test data.
Expected ROUGE scores on test set:
- ROUGE-1: 0.5-0.6
- ROUGE-2: 0.5-0.6
- ROUGE-L: 0.4-0.6

## Deployment Links
The Flask UI is deployed at - https://medical-test-summarisation-1.onrender.com/
The Fast API is deployed at - https://medical-test-summarisation.onrender.com/docs

## Limitations
- The Flan-T5 model has not been tested in real world applications.
-The model is vulnerable to generating inappropriate content or replicating inherent biases

## Model Choice
Common models that could be used for this summariasation task are BART, T5 and GPT. LLM like GPT can be fine tuned using prompting or RAG. This is a small project and can be run on a CPU so a Hugging face transformer model google/Flan-T5 seemed a good choice. This model was fine tuned to produce medical summaries.
- The Flan-T5-small model has 77M parameters. It can be easily fine tuned and run on a CPU.

## LoRA Reasoning
- LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the base model's weights and trains only smaller, low-rank adapter matrices. Its primary advantages are drastically lower GPU memory requirements, significantly faster training times, and the ability to prevent catastrophic forgetting. 
- It is ideal for training a LLM with millions of parameters
- A trained LoRA adapter is typically just a few megabytes in size (instead of gigabytes or terabytes), making it incredibly easy to store, share, and deploy multiple specialized versions of a single base model.

## Training Constraints
- Running it on a CPU can take 1 to 2 hours for training the model where there are approximately 1400 findings
- The model will struggle on very long documents. It is best suited for summariasation tasks on smaller documents.
- Flan-T5 has not been tested in real world applications.

## Improvement Suggestions
- Flan-T5 can be swapped with a paid model that is trained on medical data to give more accurate summaries
- RAG - Retrieval Augumented Generation can be used to train the model with medical data to produce better results