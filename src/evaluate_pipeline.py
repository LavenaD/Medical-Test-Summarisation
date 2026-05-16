import os

from sklearn.metrics import classification_report
import evaluate_pipeline
import evaluate

import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import display
import joblib


rouge = evaluate.load("rouge")

class EvaluatePipeline:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def __init__(self):
        pass

    def evaluate_model(self):

        X_test = pd.read_csv(os.path.join(EvaluatePipeline.BASE_DIR, "data", "processed", "test_data.csv"))["findings"]
        y_test = pd.read_csv(os.path.join(EvaluatePipeline.BASE_DIR, "data", "processed", "test_data.csv"))["impression"]

        pipeline = joblib.load(os.path.join(EvaluatePipeline.BASE_DIR, "models", "logistic_regression_pipeline.pkl"))
        label_encoder = joblib.load(os.path.join(EvaluatePipeline.BASE_DIR, "models", "label_encoder.pkl"))

        y_pred = pipeline.predict(X_test)

        # Inverse transform to get actual impression text
        y_pred_text = label_encoder.inverse_transform(y_pred)
        y_test_text = label_encoder.inverse_transform(y_test)

        print(classification_report(y_test, y_pred, zero_division=0))

        self.pipeline = pipeline

        df = pd.DataFrame(data={'predicted_summary': y_pred_text, 'actual_impressions': y_test_text})

        rouge_scores = rouge.compute(
            predictions=df["predicted_summary"],
            references=df["actual_impressions"]
        )
   
        for metric, value in rouge_scores.items():
            print(f"{metric}: {value:.4f}")

        metric_summary = {
            "ROUGE-1": rouge_scores["rouge1"],
            "ROUGE-2": rouge_scores["rouge2"],
            "ROUGE-L": rouge_scores["rougeL"]
        }



        plt.figure(figsize=(9, 5))
        plt.bar(metric_summary.keys(), metric_summary.values())
        plt.title("Summarization Evaluation Metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

        df["predicted_summary_length"] = df["predicted_summary"].apply(lambda x: len(x.split()))
        df["actual_impressions_length"] = df["actual_impressions"].apply(lambda x: len(x.split()))

        plt.figure(figsize=(8, 5))
        plt.scatter(df["actual_impressions_length"], df["predicted_summary_length"])
        plt.title("Actual Impressions Length vs Predicted Summary Length")
        plt.xlabel("Actual Impressions Length")
        plt.ylabel("Predicted Summary Length")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        display(df.head(10))