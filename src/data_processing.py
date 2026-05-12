
import os
import pandas as pd

import xml_reader
import csv_writer
import cleaning_data


class DataProcessing():
    def __init__(self):
        self.data = []
        self.df = pd.DataFrame()
        # self.previous_id = None
        # self.missing_id_files = []      
    
        # return self.write_csv_file(max_rows_per_outputfile )
    def extract (self, input_folder_path, max_rows_per_outputfile=10):
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
    
    def clean_data(self, df):
        cleaning_data_processor = cleaning_data.CleaningData()
        return cleaning_data_processor.clean(df)
    
    def write_csv_file(self, cleaned_df)-> str:
        writer = csv_writer.CsvWriter()
        max_rows_per_outputfile = cleaned_df.shape[0]
        self.data = cleaned_df.to_dict(orient='records')
        return writer.write_to_file(self.data, max_rows_per_outputfile)      
    

