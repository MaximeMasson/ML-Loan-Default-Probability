from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier

# Importing all classifiers, ensemble methods, and related components
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from ray import tune

class createPipeline:
    def __init__(self, model, params_search, composing_models=None, apply_pca = False):
        # Initialize `estimator_params` with `estimators` if `composing_models` are provided
        estimator_params = {}
        
        if composing_models:
            # Set `estimators` for VotingClassifier with provided composing models
            estimator_params = {'estimators': composing_models}
            # Set the main `params_search` for each model in composing_models
            for (name, clf) in composing_models:
                # Retrieve each composing model's configuration if it exists in `configs`
                if name in configs:
                    model_params = configs[name].params_search
                    # Prefix each parameter with model name and add to `params_search` and take whats after model__
                    for param_key, param_val in model_params.items():
                        param_key = param_key.split('model__', 1)[-1]
                        params_search[f'model__{name}__{param_key}'] = param_val
        
        # Initialize the attributes
        self.model_class = model 
        self.params_search = params_search
        
        # Construct the pipeline with optional PCA step
        pipeline_steps = [('scaler', StandardScaler())]
        
        if apply_pca:
            pipeline_steps.append(('pca', PCA()))
        
        # Add the model step
        pipeline_steps.append(('model', model(**estimator_params)))
        
        # Create the pipeline
        self.model = Pipeline(pipeline_steps)
        
    def set_params(self, **params):
        self.model = self.model.set_params(**params)

configs = {
    'logistic_regression': createPipeline(
        LogisticRegression,
        {
            'model__penalty': tune.choice(['l2', 'l1']),
            'model__C': tune.uniform(0.1, 5),
            'model__solver': tune.choice(['liblinear']),
            'model__random_state': tune.choice([42]),
        }
    ),
    'xgboost': createPipeline(
        XGBClassifier,
        {
            'model__n_estimators': tune.qrandint(50, 600, 50),
            'model__max_depth': tune.randint(1, 15),
            'model__learning_rate': tune.uniform(0.01, 0.5),
            'model__subsample': tune.uniform(0.5, 1.0),
            'model__colsample_bytree': tune.uniform(0.3, 1.0),
            'model__gamma': tune.uniform(0, 5),
            'model__min_child_weight': tune.qrandint(1, 10, 1),
            'model__random_state': tune.choice([42]),
        }
    ),
    'random_forest': createPipeline(
        RandomForestClassifier,
        {
            'model__n_estimators': tune.randint(50, 300),
            'model__max_depth': tune.randint(10, 50),
            'model__min_samples_split': tune.randint(2, 10),
            'model__min_samples_leaf': tune.randint(1, 10),
            'model__max_features': tune.choice(['sqrt', 'log2', None]),
            'model__random_state': tune.choice([42]),
        }
    ),
    # Parameters adapted from the model evaluation test
    'svc': createPipeline(
        SVC,
        {
            'model__C': tune.uniform(0.1, 2),
            'model__kernel': tune.choice(['linear']),
            'model__gamma': tune.choice(['scale', 'auto']),
            'model__random_state': tune.choice([42]),
        }
    ),
    'knn': createPipeline(
        KNeighborsClassifier,
        {
            'model__n_neighbors': tune.randint(1, 20),
            'model__weights': tune.choice(['uniform', 'distance']),
            'model__algorithm': tune.choice(['auto', 'ball_tree', 'kd_tree', 'brute']),
            'model__leaf_size': tune.randint(10, 50)
        }
    ),
    'gradient_boosting': createPipeline(
        GradientBoostingClassifier,
        {
            'model__n_estimators': tune.qrandint(50, 500, 10),
            'model__learning_rate': tune.loguniform(1e-4, 0.3),
            'model__max_depth': tune.randint(3, 10),
            'model__subsample': tune.uniform(0.5, 1.0),
            'model__random_state': tune.choice([42]),
        }
    ),  
    'gradient_boosting PCA': createPipeline(
        GradientBoostingClassifier,
        {
            'model__n_estimators': tune.qrandint(50, 500, 10),
            'model__learning_rate': tune.loguniform(1e-4, 0.3),
            'model__max_depth': tune.randint(3, 10),
            'model__subsample': tune.uniform(0.5, 1.0),
            'model__random_state': tune.choice([42]),
        },
        apply_pca = True
    ),  
    'adaboost': createPipeline(
        AdaBoostClassifier,
        {
            'model__n_estimators': tune.randint(50, 500),
            'model__learning_rate': tune.uniform(0.01, 1),
            'model__random_state': tune.choice([42]),
        }
    ),
    'gaussian_nb': createPipeline(
        GaussianNB,
        {}
    ),
    
    'mlp': createPipeline(
        MLPClassifier,
        {
            'model__hidden_layer_sizes': tune.choice([(50,), (100,), (50, 50), (100, 50)]),
            'model__activation': tune.choice(['relu', 'tanh']),
            'model__learning_rate_init': tune.loguniform(1e-4, 1e-2),
            'model__max_iter': tune.choice([200, 300, 400]),
            'model__random_state': tune.choice([42]),
        }
    )
}

# Add combinations with VotingClassifier
configs.update({
    'random_forest + xgboost + gradient_boosting': createPipeline(
        VotingClassifier, 
        {
            'model__voting': tune.choice(['hard', 'soft']),
            'model__weights': tune.choice([[1, 1, 1], 
                                           [0, 0, 1], [0, 0, 1], [0, 1, 0], 
                                           [0, 1, 1], [1, 0, 1], [1, 1, 0], 
                                           [0, 1, 2], [1, 0, 2], [2, 1, 0],
                                           [2, 1, 1], [1, 2, 1], [1, 1, 2], 
                                           [2, 2, 1], [2, 1, 2], [1, 2, 2]]),
        }, 
        [
            ('random_forest', RandomForestClassifier()),
            ('xgboost', XGBClassifier()),
            ('gradient_boosting', GradientBoostingClassifier())
        ]
    ),
})