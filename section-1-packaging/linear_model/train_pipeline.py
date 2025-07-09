import numpy as np
from config.core import config
from pipeline import price_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import mean_squared_error
from linear_model import __version__ as _version

def run_training() -> None:
    """Train the model and Log with mlflow."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("linear_model_experiment")
    with mlflow.start_run(run_name="training_run"):
    # fit model
        price_pipe.fit(X_train, y_train)
        preds = price_pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mlflow.log_param("features", config.model_config.features)
        mlflow.log_param("target", config.model_config.target)
        mlflow.log_param("alpha", config.model_config.alpha)
        mlflow.log_param("model_type", "price_pipe")
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(price_pipe, "model",registered_model_name=f"{config.app_config.pipeline_save_file}{_version}")
        # persist trained model
        save_pipeline(pipeline_to_persist=price_pipe)

if __name__ == "__main__":
    run_training()
