import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import numpy as np
import joblib
from mlProject.utils.common import save_json
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
import mlflow
# import dagshub
# dagshub.init(repo_owner='juluri.saichandu99', repo_name='MLOPs', mlflow=True)



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.scores = {"rmse": 0, "mae": 0, "r2": 0}

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    


    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        self.model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = self.model.predict(test_x)

        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
        
        # Saving metrics as local
        self.scores = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(self.config.metric_file_name), data=self.scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(self.scores)
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(self.model, "model_en", registered_model_name="ElasticNetWineModel")
            else:
                mlflow.sklearn.log_model(self.model, "model_en")

