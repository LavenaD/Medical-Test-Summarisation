import csv
import os
# from decouple import config
from datetime import datetime

# from django.conf import settings

class CsvWriter():
    def write_to_file(self, data, max_rows_per_outputfile=1000):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        date = datetime.now().strftime("%Y%m%d%H%M%S")
        # output_directory_path = os.path.join(settings.BASE_DIR, config("OUTPUT_DIRECTORY", default="OUTPUT")).replace("\\", "/")
        output_directory_path = os.path.join(base_dir, "data//processed").replace("\\", "/")
        output_file =  output_directory_path + "/" + f"processed_reports_{date}.csv"
        file_exists = os.path.exists(output_file)
        row_num = 0
        
        with open(output_file, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as csv_file:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            original_length = len(data)
            while row_num < max_rows_per_outputfile and row_num < original_length:
                # print(f"Writing row {row_num} of {original_length}")
                row = data.pop()
                if row is None:
                    break
                writer.writerow(row)
                row_num += 1
            
        return (f"Wrote {row_num} rows to {output_file}")