from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.model_evaluation import model_evaluation

@pipeline
def training_pipeline(data_path: str) -> None:
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    trained_model = train_model(X_train, X_test, y_train, y_test)
    mse, r2, rmse = model_evaluation(trained_model, X_test, y_test)
