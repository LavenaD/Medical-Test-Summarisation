import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import evaluate_pipeline
import joblib

import data_processing



class Train():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def __init__(self):
        self.pipeline = None
        
    def write_train_test_csv(self, X_train, X_test, y_train, y_test):
        train_df = pd.DataFrame({"findings": X_train, "impression": y_train})
        test_df = pd.DataFrame({"findings": X_test, "impression": y_test})

        output_directory_path = os.path.join(Train.BASE_DIR, "data//processed").replace("\\", "/")
        train_df.to_csv(output_directory_path + "/" + f"train_data.csv", index=False)
        test_df.to_csv(output_directory_path + "/" + f"test_data.csv", index=False)
        
        return 1

    def train_model(self, df):
        print("Training model...")
        TARGET_COL = "impression"
               
        label_encoder = LabelEncoder()
        df["target"] = label_encoder.fit_transform(df[TARGET_COL])
        X_train, X_test, y_train, y_test = train_test_split(df["findings"], df["target"], test_size=0.2, random_state=42)
        
        # -------------------------------------------------
        # 1. Save the training set and test set to CSV files
        # ------------------------------------------------------ 
        self.write_train_test_csv(X_train, X_test, y_train, y_test)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer( stop_words="english",
            lowercase=True,
            max_features=10000,
            ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])
        pipeline.fit(X_train, y_train)


        # -------------------------------------------------
        # 2. Save the trained model and label encoder
        # ------------------------------------------------------       
        output_directory_path = os.path.join(Train.BASE_DIR, "models").replace("\\", "/")
        joblib.dump(pipeline, os.path.join(output_directory_path, "logistic_regression_pipeline.pkl"))
        joblib.dump(label_encoder, os.path.join(output_directory_path, "label_encoder.pkl"))
  
        return pipeline
    
    # def run_predictions(self):
        
    #     # 9. Predict on new text
    #     new_finding = ["Lungs are clear bilaterally. There is no focal consolidation, pleural effusion, or pneumothoraces. Cardiomediastinal silhouette is within normal limits. XXXX are unremarkable.",
    #                     "Cardiomediastinal silhouette demonstrates normal heart size with tortuosity and atherosclerosis of the thoracic aorta. No focal consolidation, pneumothorax, or pleural effusion. No acute bony abnormality identified. Multilevel degenerative disc disease of the thoracic spine noted.",
    #                     "Calcified mediastinal XXXX. No focal areas of consolidation. Heart size within normal limits. No pleural effusions. No evidence of pneumothorax. Degenerative changes thoracic spine.",
    #                     "Sternotomy sutures and bypass grafts have been placed in the interval. Both lungs remain clear and expanded with no infiltrates. Pulmonary XXXX are normal.",
    #                     "The lungs are clear. The cardiomediastinal silhouette is within normal limits. No pneumothorax or pleural effusion.",
    #                     "Heart size normal. Lungs are clear. XXXX are normal. No pneumonia, effusions, edema, pneumothorax, adenopathy, nodules or masses.",
    #                     "There are bilateral pulmonary nodules whose appearances suggest metastatic disease to lungs. In the right lung, there is a 1.9 x 2.1 cm nodule overlying the posterior right 6th rib. There is a 1.0 x 1.2 cm nodule XXXX above this in the interspace between the posterior 5th and 6th ribs on the right. There is a 1.0 x 1.1 cm nodule projecting through the left 9th and 10th interspaces on the PA view. If not already performed, contrast-enhanced XXXX would be XXXX suited to evaluate these findings. There are no focal airspace opacities to suggest pneumonia. To the stomach contours appear grossly clear. Heart size and pulmonary XXXX appear normal. There are left-sided axillary clips. There is a right internal jugular central catheter, the distal tip in right atrium.",
    #                     "The heart size is upper limits of normal. The pulmonary XXXX and mediastinum are within normal limits. There is no pleural effusion or pneumothorax. There is mild streaky perihilar opacity without confluent airspace opacity to suggest a bacterial pneumonia.",
    #                     "Heart size within normal limits. No focal airspace consolidations. No pneumothorax or effusions.",
    #                     "Normal heart and mediastinum. Clear lungs. Trachea is midline. No pneumothorax. No pleural effusion. Radiopaque foreign body overlying left chest.",
    #                     "The cardiac contours are normal. The lungs are clear. Thoracic spondylosis. Prior cholecystectomy",
    #                     "Lungs are clear. No focal infiltrate. No pleural effusion or pneumothorax. Normal cardiomediastinal silhouette.",
    #                     "The heart and lungs have XXXX XXXX in the interval. Both lungs are clear and expanded. Heart and mediastinum normal.",
    #                     "The lungs are clear. There is no focal airspace consolidation. No pleural effusion or pneumothorax. Heart size and mediastinal contour are normal.",
    #                     "The cardiomediastinal silhouette is within normal limits for appearance. There are low lung volumes with bronchovascular crowding and scattered XXXX opacities in the bilateral lung bases. No focal areas of pulmonary consolidation. No pneumothorax. No large pleural effusion. No acute, displaced rib fractures identified.",
    #                     "Normal heart size and mediastinal contours. There are reticular opacities in the medial right middle lobe with tubular airway ectasia which obscures the right heart XXXX. This was present previously and is most compatible with bronchiectasis. There is no XXXX focal airspace disease. No pneumothorax or pleural effusion. Unremarkable XXXX.",
    #                     "Lungs are clear. No focal airspace consolidation. No pleural effusion or pneumothorax. Normal cardiomediastinal silhouette. There are postoperative changes of cervical spine fusion.",
    #                     "Heart size and mediastinal contours appear within normal limits. Pulmonary vascularity is within normal limits. No focal consolidation, suspicious pulmonary opacity, pneumothorax or definite pleural effusion. Visualized osseous structures appear intact.",
    #                     "The heart size is upper limits of normal. The pulmonary XXXX and mediastinum are within normal limits. There is no pleural effusion or pneumothorax. There is mild streaky perihilar opacity without confluent airspace opacity to suggest a bacterial pneumonia."]
    #     for finding in new_finding:
    #         print(f"\nFinding: {finding}")
    #         finding = cleaning_data.CleaningData().clean_text(finding)
    #         prediction = self.pipeline.predict([finding])
    #         label_encoder = LabelEncoder()
    #         predicted_label = label_encoder.inverse_transform(prediction)
    #         print(f"\nPredicted Label: {predicted_label}")

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

        trainer = Train()
        pipeline = trainer.train_model(cleaned_df)

        evaluator = evaluate_pipeline.EvaluatePipeline()
        evaluator.evaluate_model()

    except Exception as e:
        print(f"An error occurred: {e}")