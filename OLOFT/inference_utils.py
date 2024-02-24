import numpy as np
import json
from sklearn.base import BaseEstimator
from mapie.classification import MapieClassifier

class autogluon_wrapper(BaseEstimator):
    def __init__(self, model):
        self.model = model
        self.fitted = True
        self.classes_ = model.class_labels

    def fit(self, X, y, sample_weight=None):
        return True

    def predict(self, X):
        return self.model.predict(X).to_numpy()
    
    def predict_proba(self, X):
        return self.model.predict_proba(X).to_numpy()
    

def get_conformal_threshold(model, X_calibiration, y_calibiration, alpha=0.1):

    wrapped_model = autogluon_wrapper(model)
    conformal_predictor = MapieClassifier(estimator=wrapped_model, cv="prefit", method="cumulated_score") # type: ignore
    conformal_predictor.fit(X_calibiration, y_calibiration)
    conformal_predictor.predict(X_calibiration, alpha=alpha)

    return conformal_predictor.quantiles_[0]

def get_conservative_predictions(model, X, threshold, top_k = 2):

    pred_pci = model.predict_proba(X).to_numpy()
    pred_pci_l = model.predict(X).to_numpy()
    labels = model.class_labels

    top_k_idx = np.argsort(-pred_pci, axis=1)[:, :top_k]
    top_k_labels = labels[top_k_idx]

    pci_p_final = np.where(pred_pci.max(axis=1) > threshold, pred_pci_l, top_k_labels.min(1))

    return pci_p_final

def save_json(test_image_names, predictions, target_json='results.json'):
    results = []
    for i in range(len(test_image_names)):
        results.append({test_image_names[i]: predictions[i]})

    with open(target_json, 'w') as f:
        json.dump(results, f)