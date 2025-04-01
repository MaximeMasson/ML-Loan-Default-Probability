from _Model_Pipeline_class import configs

from tune_sklearn import TuneSearchCV
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split

import pandas as pd
import time

class modelComparator:
    def __init__(self, df: pd.DataFrame):
        # Initialize with the input DataFrame and perform train-test split
        self.dataframe = df
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df.drop(columns=['target']), df['target'], test_size=0.2, random_state=42, stratify=df['target']
        )
        self.metrics_dict = {}  # Dictionary to store evaluation metrics

    # The function to perform cross-validation and hyperparameter search
    def performCV(self, config, kfold, n_trials):
        # If no hyperparameters to search, directly fit the model
        if not config.params_search:
            modelCV = config.model
            modelCV.fit(self.X_train, self.y_train)
            self.best_params = {}  # No hyperparameters were optimized
        else:
            # Use TuneSearchCV with Optuna for hyperparameter tuning
            modelCV = TuneSearchCV(
                config.model,
                config.params_search, 
                search_optimization="optuna", 
                scoring=make_scorer(roc_auc_score),
                cv=kfold,
                verbose=0,
                n_jobs=-1,
                n_trials=n_trials,
                # max_iters=1500
            )
            modelCV.fit(self.X_train, self.y_train)
            self.best_params = modelCV.best_params_  # Save best hyperparameters found

    # The function to perform the test
    def performTest(self, config, kfold, n_trials):
        # Perform cross-validation and hyperparameter search
        self.performCV(config, kfold, n_trials)
        
        # Set up and retrain the best model with optimized parameters
        self.best_model = config.model
        self.best_model.set_params(**self.best_params)
        self.best_model.fit(self.X_train, self.y_train)
        
        # Make predictions on the test set
        self.y_pred = self.best_model.predict(self.X_test)


        # Calculate and store evaluation metrics
        recall = recall_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        auc_roc = roc_auc_score(self.y_test, self.y_pred)

        # Add metrics to the dictionary for output
        self.metrics_dict = {
            'Best Params': self.best_params,
            'Recall': recall,
            'Precision': precision,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'AUC ROC': auc_roc
        }
    
    # The function to compare multiple models     
    def compare_models(self, configs_list: list, kfolds: int = 5, n_trials: int = 10):
        results_df = pd.DataFrame()  # DataFrame to store results for each model
        
        for model_name in configs_list:
            print(f"Testing model: {model_name}")
            try:
                # Measure model evaluation time
                start_time = time.time()
                
                # Retrieve the specific model configuration
                config = configs[model_name]
                self.performTest(config, kfolds, n_trials)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Add model name and processing time to metrics
                self.metrics_dict['Model'] = model_name
                self.metrics_dict['Processing Time (s)'] = processing_time
                
                print(self.metrics_dict, flush=True)
                # Append metrics to results DataFrame
                results_df = results_df.append(self.metrics_dict, ignore_index=True)
            except Exception as e:
                print(f"Error testing model {model_name}: {e}")
        
        self.results_df = results_df  # Save the results as an attribute
        return self.results_df  # Return the results DataFrame for analysis
