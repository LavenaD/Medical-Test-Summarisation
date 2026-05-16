import os
import pandas as pd
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from IPython.display import display


def run_evaluation_job():
    
    # load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    # Hugging Face repo
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_directory_path = os.path.join(base_dir, "models//compressed-artifacts-google").replace("\\", "/")
    
    model_path = os.path.join(output_directory_path, "medical_summarizer_peft")
    # attach LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # print(datetime.datetime.now(), "Model and tokenizer loaded for job:", job_id)

    device = torch.device("cpu")
    model.to(device)

    # print(datetime.datetime.now(), "Model moved to device for job:", job_id)

    # load data
    csv_path = os.path.join(base_dir, "data", "processed", "test_data.csv")
    test_df = pd.read_csv(csv_path)

    inputs = test_df["findings"].tolist()
    references = test_df["impression"].tolist()

    # predict
    predictions = []

    batch_size = 8  # try 4, 8, or 16 depending on memory
    predictions = []

    model.eval()

    for start_idx in range(0, len(inputs), batch_size):
        end_idx = min(start_idx + batch_size, len(inputs))
        batch_texts = inputs[start_idx:end_idx]

        print(
            f"Generating output for batch {start_idx}:{end_idx} of {len(inputs)}"
        )

        prompted_batch = [
            "Summarize this medical condition: " + str(text)
            for text in batch_texts
        ]

        tokens = tokenizer(
            prompted_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **tokens,
                max_length=256,
                num_beams=5
            )

        batch_predictions = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        predictions.extend(batch_predictions)

    # print(classification_report(y_test, y_pred, zero_division=0))

    rouge = evaluate.load("rouge")

    print(
            f"Calculating ROUGE scores for {len(predictions)} predictions and {len(references)} references"
        )

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references
    )

    for metric, value in rouge_scores.items():
        print(f"{metric}: {value:.4f}")

    metric_summary = {
        "ROUGE-1": rouge_scores["rouge1"],
        "ROUGE-2": rouge_scores["rouge2"],
        "ROUGE-L": rouge_scores["rougeL"]
    }

    results = pd.DataFrame({
        "predictions": predictions,
        "references": references
    })

    plt.figure(figsize=(9, 5))
    plt.bar(metric_summary.keys(), metric_summary.values())
    plt.title("Summarization Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    results["predicted_summary_length"] = results["predictions"].apply(lambda x: len(x.split()))
    results["actual_impressions_length"] = results["references"].apply(lambda x: len(x.split()))

    plt.figure(figsize=(8, 5))
    plt.scatter(results["actual_impressions_length"], results["predicted_summary_length"])
    plt.title("Actual Impressions Length vs Predicted Summary Length")
    plt.xlabel("Actual Impressions Length")
    plt.ylabel("Predicted Summary Length")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    display(results.head(10))

    return results


