
import os
import pandas as pd

import xml_reader
import csv_writer
import cleaning_data
import train
import evaluate_pipeline

class DataProcessing():
    def __init__(self):
        self.data = []
        self.df = pd.DataFrame()
        # self.previous_id = None
        # self.missing_id_files = []      
    
        # return self.write_csv_file(max_rows_per_outputfile )
    def read_xml_files(self, input_folder_path, max_rows_per_outputfile=10):
        print(f"Processing files in directory: {input_folder_path}")
        if not os.path.exists(input_folder_path):
            print(f"Input folder path '{input_folder_path}' does not exist.")
            return pd.DataFrame()
        for file in os.listdir(input_folder_path):
            if file.endswith(".xml"):
                file_path = os.path.join(input_folder_path, file)
        
                data_dict = xml_reader.XmlReader().read_file(file_path)
                if data_dict is not None:
                    self.data.append(data_dict) 
        return pd.DataFrame(self.data)
    
    def write_csv_file(self, max_rows_per_outputfile)-> str:
        writer = csv_writer.CsvWriter()
        return writer.write_to_file(self.data, max_rows_per_outputfile)    
    
if __name__ == "__main__":
    try:
        input_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data\\raw").replace("\\", "/")
        print(f"Input folder path: {input_folder_path}")

        reader = DataProcessing()
        df = reader.read_xml_files(input_folder_path, max_rows_per_outputfile=100)
        print(f"Number of records read: {len(df)}")

        cleaned_data_processor = cleaning_data.CleaningData()
        cleaned_df = cleaned_data_processor.cleaning_data(df)
        print(f"Number of records after cleaning: {len(cleaned_df)}")

        trainer = train.Train()
        y_pred_text, y_test_text, y_test, y_pred, pipeline, label_encoder = trainer.train_model(cleaned_df)

        evaluator = evaluate_pipeline.EvaluatePipeline()
        evaluator.evaluate_model(y_test_text, y_test, y_pred_text, y_pred)

    except Exception as e:
        print(f"An error occurred: {e}")
