import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report


TRAINING_DATA_PATH = "..//data//training_data.pkl"

# code taken from view.py
def load_tensors(path, seed=42, test_size=0.2):
    # load pkl file
    with open(path, 'rb') as f:
        training_data = pickle.load(f)
        
    X_user = []
    X_game = []
    X_cross = []
    y = []

    for x in training_data:
        # User features
        user_feats = []
        for k, v in x['user_feature'].items():
            if isinstance(v, np.ndarray):
                user_feats.extend(v.tolist())
            else:
                user_feats.append(v)
        X_user.append(user_feats)

        # Game features
        game_feats = []
        for k, v in x['game_feature'].items():
            if isinstance(v, np.ndarray):
                game_feats.extend(v.tolist())
            else:
                game_feats.append(v)
        X_game.append(game_feats)

        # Cross features
        cross_feats = []
        for k, v in x['cross_feature'].items():
            cross_feats.append(v)
        X_cross.append(cross_feats)

        # Label
        y.append(x['label'])

    X_user = np.array(X_user, dtype=np.float32)
    X_game = np.array(X_game, dtype=np.float32)
    X_cross = np.array(X_cross, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X = np.concatenate([X_user, X_game, X_cross], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=y
    )
    
    # return training and testing data
    return X_train, X_test, y_train, y_test

# init tensors
x_train, x_test, y_train, y_test = load_tensors(TRAINING_DATA_PATH)

# optuna hyperparam tuning
def objective(trial: optuna.Trial) -> float:

    # hyper param search space
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 5000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.001, 0.75, log=True),
    }
    
    model = xgb.XGBClassifier(
        
        **params,
        device='cuda',
        use_label_encoder=False,
    )
    
    model.fit(
        x_train,
        y_train,
        verbose=False
    )
    
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    
    # return value optuna will try to maximize
    return macro_f1
    

# script entry point
if __name__ == '__main__':
    
    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_recommender",
    )   
    
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    print("Best AUC:", study.best_value)
    print("Best params:", study.best_params)
    
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
    })
    
    best_model = xgb.XGBClassifier(
        **best_params,
        device='cuda',
        use_label_encoder=False,
    )
    
    best_model.fit(
        x_train,
        y_train,
        verbose=False,
    )
    
    # eval on test set
    y_test_proba = best_model.predict_proba(x_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, y_test_proba)
    test_f1 = f1_score(y_test, y_test_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test F1:  {test_f1:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_test_pred))
    