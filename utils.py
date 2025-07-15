from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_metrics(y_true, y_pred):
    # Calculating F1 scores for each class
    f1_scores_per_class = f1_score(y_true, y_pred, average=None)
    
    dict_metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Macro Recall': recall_score(y_true, y_pred, average='weighted'),
        'Macro Precision': precision_score(y_true, y_pred, average='weighted',zero_division=0),
        'Macro F1': f1_score(y_true, y_pred, average='weighted',zero_division=0),
        'F1 Scores per Class': f1_scores_per_class
    }
    return dict_metrics

def preprocessor(df, numeric_features, categorical_features):
    """
    Preprocesses the dataset by applying StandardScaler to numeric features
    and OneHotEncoder to categorical features using ColumnTransformer.

    Parameters:
    - df: pd.DataFrame, the input dataset
    - numeric_features: list, names of numeric columns
    - categorical_features: list, names of categorical columns

    Returns:
    - pd.DataFrame, the transformed dataset with appropriate column names
    """
    # Define pipelines for numeric and categorical features
    num_pipeline = Pipeline([
        ('num_features', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('cat_features', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create the ColumnTransformer
    column_transformer = ColumnTransformer(transformers=[
        ('num_trans', num_pipeline, numeric_features),
        ('cat_trans', cat_pipeline, categorical_features)
    ])

    # Fit and transform the data
    transformed_data = column_transformer.fit_transform(df)

    # Get the new column names
    encoded_feature_names = column_transformer.named_transformers_['cat_trans'].get_feature_names_out(categorical_features)
    new_column_names = list(numeric_features) + list(encoded_feature_names)

    # Convert the transformed data back to a DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=new_column_names)

    return transformed_df



