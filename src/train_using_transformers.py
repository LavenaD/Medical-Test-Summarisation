import os
import numpy as np
import evaluate
import data_processing

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
class TrainUsingTransformers:   
        # ==================================================
        # 1. CONFIG
        # ==================================================

        MODEL_NAME = "t5-small"
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        OUTPUT_DIRECTORY_PATH = os.path.join(BASE_DIR, "data//processed").replace("\\", "/")
        DATA_PATH = os.path.join(OUTPUT_DIRECTORY_PATH, "summaries.csv").replace("\\", "/")
        OUTPUT_DIR = os.path.join(BASE_DIR, "models/t5_summariser").replace("\\", "/")

        TEXT_COLUMN = "findings"
        SUMMARY_COLUMN = "impression"

        MAX_INPUT_LENGTH = 512
        MAX_TARGET_LENGTH = 128

        def __init__(self):

            # ==================================================
            # 2. LOAD DATASET
            # ==================================================

            dataset = load_dataset("csv", data_files=self.DATA_PATH)

            self.dataset = dataset["train"].train_test_split(
                test_size=0.2,
                seed=42
            )

            self.train_dataset = self.dataset["train"]
            self.eval_dataset = self.dataset["test"]

            # ==================================================
            # 3. LOAD TOKENIZER AND MODEL
            # ==================================================

            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)

            # ==================================================
            # 4. PREPROCESS DATA
            # ==================================================

        def preprocess_function(self):
            inputs = [
                "summarize: " + text
                for text in  self.train_dataset[self.TEXT_COLUMN]
            ]

            model_inputs = self.tokenizer(
                inputs,
                max_length=self.MAX_INPUT_LENGTH,
                truncation=True
            )

            labels = self.tokenizer(
                text_target= self.train_dataset[self.SUMMARY_COLUMN],
                max_length=self.MAX_TARGET_LENGTH,
                truncation=True
            )

            model_inputs["labels"] = labels["input_ids"]

            


            self.tokenized_train = self.train_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=self.train_dataset.column_names
            )

            self.tokenized_eval = self.eval_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=self.eval_dataset.column_names
            )

            print(model_inputs["predictions"])
            return model_inputs

            # ==================================================
            # 5. METRICS
            # ==================================================

            

        def compute_metrics(self, eval_pred):
            rouge = evaluate.load("rouge")
            predictions, labels = eval_pred

            decoded_preds = self.tokenizer.batch_decode(
                predictions,
                skip_special_tokens=True
            )

            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

            decoded_labels = self.tokenizer.batch_decode(
                labels,
                skip_special_tokens=True
            )

            result = rouge.compute(
                predictions=decoded_preds,
                references=decoded_labels
            )

            results = {
                "rouge1": result["rouge1"],
                "rouge2": result["rouge2"],
                "rougeL": result["rougeL"],
                "rougeLsum": result["rougeLsum"]
            }

            # ==================================================
            # 6. TRAINING SETTINGS
            # ==================================================

            training_args = Seq2SeqTrainingArguments(
                output_dir=self.OUTPUT_DIR,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                weight_decay=0.01,
                save_total_limit=2,
                num_train_epochs=3,
                predict_with_generate=True,
                fp16=False,  # Set True only if you have a compatible NVIDIA GPU
                logging_dir="./logs",
                logging_steps=50,
                report_to="none"
            )

            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model
            )

            # ==================================================
            # 7. TRAINER
            # ==================================================

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.tokenized_train,
                eval_dataset=self.tokenized_eval,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics
            )

            # ==================================================
            # 8. TRAIN
            # ==================================================

            trainer.train()

            # ==================================================
            # 9. SAVE MODEL
            # ==================================================

            trainer.save_model(self.OUTPUT_DIR)
            self.tokenizer.save_pretrained(self.OUTPUT_DIR)

            print(f"Model saved to: {self.OUTPUT_DIR}")

            print(f"Training completed successfully.{results}")

if __name__ == "__main__":
    try:
        # input_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data\\raw").replace("\\", "/")
        # print(f"Input folder path: {input_folder_path}")

        # data_processing = data_processing.DataProcessing()
        # df = data_processing.extract(input_folder_path, max_rows_per_outputfile=100)
        # print(f"Number of records read: {len(df)}")

        # cleaned_df = data_processing.clean_data(df)
        # print(f"Number of records after cleaning: {len(cleaned_df)}")

        # data_processing.write_csv_file(cleaned_df)

        trainer = TrainUsingTransformers()
        pipeline = trainer.preprocess_function()

        final_output = trainer.compute_metrics(pipeline)

    except Exception as e:
        print(f"An error occurred: {e}")