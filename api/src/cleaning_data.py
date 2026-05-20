import pandas as pd
import re

class CleaningData():
    TEXT_COL = "findings"
    TARGET_COL = "impression"
    def __init__(self):
        self.cleaned_data = pd.DataFrame()

    def remove_duplicates(self, data):

        data.dropna(subset=[CleaningData.TEXT_COL, CleaningData.TARGET_COL], inplace=True)
        data.drop_duplicates(inplace=True)
        return data

    def clean_text(self, text):
        text = str(text).lower()
        text = text.replace("xxxx", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()    
        return text

    def clean_labels(self, labels):
        label_mapping = {
            'no acute cardiopulmonary abnormalities.': 'no acute cardiopulmonary abnormality.',
            'no acute cardiopulmonary findings.': 'no acute cardiopulmonary abnormality.',
            'no acute pulmonary abnormality.': 'no acute pulmonary disease.',
            'no active disease.': 'no acute disease.',
            'no evidence of active disease.': 'no acute disease.',
            '1. no evidence of active disease.': 'no acute disease.',
            '1. no acute radiographic cardiopulmonary process.': 'no acute cardiopulmonary abnormality.',
            '1. no acute cardiopulmonary disease.': 'no acute cardiopulmonary abnormality.',
            '1. no acute cardiopulmonary abnormality.': 'no acute cardiopulmonary abnormality.',
            '1. no acute pulmonary abnormality.': 'no acute pulmonary disease.',
            
        }

        return labels.replace(label_mapping)

    def clean(self, df)-> pd.DataFrame:
        try:
            if df.empty:
                print("The input DataFrame is empty. No data to clean.")
                return pd.DataFrame()
            self.remove_duplicates(df)
            df[CleaningData.TEXT_COL] = df[CleaningData.TEXT_COL].apply(self.clean_text)
            df[CleaningData.TARGET_COL] = df[CleaningData.TARGET_COL].astype(str).str.lower().str.strip()
            df[CleaningData.TARGET_COL] = self.clean_labels(df[CleaningData.TARGET_COL])

            # This helps avoid train/test errors when a label appears only once
            label_counts = df[CleaningData.TARGET_COL].value_counts()
            valid_labels = label_counts[label_counts >= 2].index
            df = df[df[CleaningData.TARGET_COL].isin(valid_labels)]

            print(df[CleaningData.TARGET_COL].value_counts().head(20))


            return df
        except Exception as e:
            print(f"An error occurred during data cleaning: {e}")
            return pd.DataFrame()