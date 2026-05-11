from sklearn.metrics import classification_report
import evaluate_pipeline
import evaluate

rouge = evaluate.load("rouge")

class EvaluatePipeline:
    def __init__(self):
        pass

    def evaluate_model(self, y_test_text, y_test, y_pred_text, y_pred):
        print(classification_report(y_test, y_pred, zero_division=0))

        results = rouge.compute(
            predictions=y_pred_text,
            references=y_test_text
        )

        print(results)
