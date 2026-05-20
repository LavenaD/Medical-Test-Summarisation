import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import src.config

def run_inference(medical_text):
    """
    Run inference on medical text to generate a summary.
    
    Args:
        medical_text (str or list): Single text or list of texts to summarize
        batch_size (int): Batch size for processing multiple texts
    
    Returns:
        str or list: Generated summary/summaries
    """
    
    # Set model path
    model_path = os.path.join(src.config.MODEL_DIRECTORY_PATH, "medical_t5_v1")
    
    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(src.config.MODEL_NAME)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set device
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    predictions = []

    # Add prefix to each text
    prompted_batch = [
        "summarize: " + str(medical_text)
    ]
    
    # Tokenize
    tokens = tokenizer(
        prompted_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(device)
    
    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(
            **tokens,
            max_length=256,
            num_beams=5,
            early_stopping=True
        )
    
    # Decode predictions
    batch_predictions = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )
    
    predictions.extend(batch_predictions)
    
    # Return single result if input was single text
    return predictions[0]

if __name__ == "__main__":
    sample_text = "heart size is normal the lungs are clear no pneumothorax or pleural effusion"
    summary = run_inference(sample_text)
    print("Generated Summary:", summary)