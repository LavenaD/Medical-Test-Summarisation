
import os
import pandas as pd

from transformers import Trainer, TrainingArguments
from datasets import Dataset
# from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# import torch
# import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model

import data_processing
import config
import evaluate_trainmodel_v2

class TrainT5Small:
    def __init__(self):
        self.model_name = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def write_train_test_csv(self, X_train, X_test, y_train, y_test):
        train_df = pd.DataFrame({config.TEXT_COL: X_train, config.TARGET_COL: y_train})
        test_df = pd.DataFrame({config.TEXT_COL: X_test, config.TARGET_COL: y_test})

        train_df.to_csv(config.DATA_DIRECTORY_PATH + "/" + "train_data.csv", index=False)
        test_df.to_csv(config.DATA_DIRECTORY_PATH + "/" + "test_data.csv", index=False)
        
        return 1

    def tokenize_function(self, df):
        # When you use text_target= in the tokenizer, it creates a column named "labels" (not "impression"). 
        # The Trainer expects this exact column name for seq2seq models
        return self.tokenizer(
                [str(f) for f in df[config.TEXT_COL]],
                text_target=[str(l) for l in df[config.TARGET_COL]],
                truncation=True
            )

    def train(self, df):
        df[config.TEXT_COL] = "Summarize this medical condition: " + df[config.TEXT_COL]
        df['labels'] = df[config.TARGET_COL]

        # Split the data into train and test
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
        self.write_train_test_csv(train_df[config.TEXT_COL], test_df[config.TEXT_COL], train_df[config.TARGET_COL], test_df[config.TARGET_COL])

        dataset = Dataset.from_pandas(train_df)

        # Split the dataset into train and test sets using the Dataset's own split method
        split_datasets = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_datasets["train"]
        eval_dataset = split_datasets["test"]           

        # Use Lora to train the model
        lora_config = LoraConfig(
            r=14,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )

        self.model = get_peft_model(self.model, lora_config)

        # Set the training arguments
        training_args = TrainingArguments(
            output_dir="./medical_model",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=config.NUM_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch"
        )
        # Apply tokenization to the datasets
        tokenized_train_dataset = train_dataset.map(self.tokenize_function, batched=True, remove_columns=[config.TEXT_COL])
        tokenized_eval_dataset = eval_dataset.map(self.tokenize_function, batched=True, remove_columns=[config.TEXT_COL])

        # Make sure the 'labels' column is correctly set for training
        tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        tokenized_eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset
        )

        trainer.train()
        # return trainer

        # # 8. Save the model
        self.model.save_pretrained(config.MODEL_DIRECTORY_PATH + "/" + "medical_summarizer_peft")
        self.tokenizer.save_pretrained(config.MODEL_DIRECTORY_PATH + "/" + "medical_summarizer_peft")

# folder_to_compress = "/content/medical_model/compressed-artifacts-g5/medical_summarizer_peft"
# output_zip_name = "medical_summarizer_model_g5.zip"

# # Create a ZipFile object
# with zipfile.ZipFile(output_zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
#     for root, dirs, files_in_folder in os.walk(folder_to_compress):
#         for file in files_in_folder:
#             file_path = os.path.join(root, file)
#             # Add file to zip, preserving directory structure relative to folder_to_compress
#             zipf.write(file_path, os.path.relpath(file_path, folder_to_compress))

# print(f"'{folder_to_compress}' successfully compressed to '{output_zip_name}'")

# # Download the zip file
# files.download(output_zip_name)

if __name__ == "__main__":
    try:
        input_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data\\raw").replace("\\", "/")
        print(f"Input folder path: {input_folder_path}")
        print(config.TEXT_COL)

        data_processing = data_processing.DataProcessing()
        df = data_processing.extract(input_folder_path, max_rows_per_outputfile=100)
        print(f"Number of records read: {len(df)}")

        cleaned_df = data_processing.clean_data(df)
        print(f"Number of records after cleaning: {len(cleaned_df)}")

        data_processing.write_csv_file(cleaned_df)

        trainer = TrainT5Small()
        pipeline = trainer.train(cleaned_df)

        evaluation_results = evaluate_trainmodel_v2.run_evaluation_job()
        print(evaluation_results)

    except Exception as e:
        print(f"An error occurred: {e}")