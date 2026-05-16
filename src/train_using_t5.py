import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from transformers import Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import data_processing

class Train_T5:
    def __init__(self):
        TARGET_COL = "impression"
        TEXT_COL = "findings"
        SUMMARISE_COL = "summerise_findings"

        try:
            self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        except Exception as e:
            print(f"Error occurred while initializing BASE_DIR: {e}")

        try:
            self.OUTPUT_DIRECTORY_PATH = os.path.join(self.BASE_DIR, "data//processed").replace("\\", "/")
        except Exception as e:
            print(f"Error occurred while initializing OUTPUT_DIRECTORY_PATH: {e}")

        try:
            self.DATA_PATH = os.path.join(self.OUTPUT_DIRECTORY_PATH, "summaries.csv").replace("\\", "/")
        except Exception as e:
            print(f"Error occurred while initializing DATA_PATH: {e}")

        try:
            self.OUTPUT_DIR = os.path.join(self.BASE_DIR, "models/t5_summariser").replace("\\", "/")
        except Exception as e:
            print(f"Error occurred while initializing OUTPUT_DIR: {e}")

        return

    def train(self, dataset):
        try:
            
            MODEL_NAME = "t5-small"
            MAX_INPUT_LENGTH = 512
            MAX_TARGET_LENGTH = 128

            dataset[self.SUMMARISE_COL] = "Summarize this medical condition: " + dataset[self.TEXT_COL]
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

            lora_config = LoraConfig(
                r=14,
                lora_alpha=16,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )

            model = get_peft_model(model, lora_config)

            training_args = TrainingArguments(
                    output_dir=self.OUTPUT_DIR + "/medical_model",
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    num_train_epochs=4,
                    learning_rate=8.8e-5,
                    logging_steps=100,
                    eval_strategy="epoch",
                    save_strategy="epoch"
                )
            
            # Convert pandas DataFrame to Hugging Face Dataset
            dataset = Dataset.from_pandas(dataset)

            # Split the dataset into train and test sets using the Dataset's own split method
            split_datasets = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_datasets["train"]
            test_dataset = split_datasets["test"]

            # Apply tokenization to the datasets
            tokenized_train_dataset = train_dataset.map(self.tokenize_function, batched=True)
            tokenized_test_dataset = test_dataset.map(self.tokenize_function, batched=True)

            # Remove the original text columns as they are no longer needed
            tokenized_train_dataset = tokenized_train_dataset.remove_columns([self.TEXT_COL])
            tokenized_test_dataset = tokenized_test_dataset.remove_columns([self.TEXT_COL])

            # Make sure the 'labels' column is correctly set for training
            tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
            tokenized_test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_test_dataset
            )

            trainer.train()


            model.save_pretrained( os.path.join(self.BASE_DIR, "models/compressed-artifacts-t5").replace("\\", "/"))
            self.tokenizer.save_pretrained( os.path.join(self.BASE_DIR, "models/compressed-artifacts-t5").replace("\\", "/"))
            return trainer
        except Exception as e:
            print(f"Error occurred while loading dataset: {e}")

    
    # Define the tokenization function
    def tokenize_function(self,dataset):
    # return tokenizer(examples["findings"],
    #                  text_target=examples["labels"],
    #                  truncation=True)
        return self.tokenizer(
                [str(f) for f in dataset[self.SUMMARISE_COL]],
                text_target=[str(_) for _ in dataset[self.TARGET_COL]],
                truncation=True
            )
if __name__ == "__main__":
    try:

        input_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data\\raw").replace("\\", "/")
        print(f"Input folder path: {input_folder_path}")

        data_processing = data_processing.DataProcessing()
        df = data_processing.extract(input_folder_path, max_rows_per_outputfile=100)
        print(f"Number of records read: {len(df)}")

        cleaned_df = data_processing.clean_data(df)
        print(f"Number of records after cleaning: {len(cleaned_df)}")

        data_processing.write_csv_file(cleaned_df)


        trainer = Train_T5()
        pipeline = trainer.train(cleaned_df)

        # evaluator = evaluate_pipeline.EvaluatePipeline()
        # evaluator.evaluate_model()

    except Exception as e:
        print(f"An error occurred: {e}")