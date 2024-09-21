import joblib 
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        # Use Path.resolve() to ensure absolute path
        model_path = Path('artifacts/model_trainer/model.joblib').resolve()
        self.model = joblib.load(model_path)
    
    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction

