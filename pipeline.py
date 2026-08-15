import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def to_str_func(X):
    return X.astype(str)

class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        Q1, Q3 = X.quantile(0.25), X.quantile(0.75)
        IQR = Q3 - Q1
        self.lower_ = Q1 - self.factor * IQR
        self.upper_ = Q3 + self.factor * IQR
        return self
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            X.iloc[:, i] = X.iloc[:, i].clip(self.lower_.iloc[i], self.upper_.iloc[i])
        return X.values

class OBDIIDeploymentPipeline:
    def __init__(self, preprocessor, vt, sel_idx, selected_feature_names, model, feature_cols):
        self.preprocessor           = preprocessor
        self.vt                     = vt
        self.sel_idx                = sel_idx
        self.selected_feature_names = selected_feature_names
        self.model                  = model
        self.feature_cols           = feature_cols

    def _prep(self, X_raw):
        if isinstance(X_raw, dict):
            X_raw = pd.DataFrame([X_raw])
        elif isinstance(X_raw, list):
            X_raw = pd.DataFrame(X_raw)
        X_raw = X_raw.copy()
        for c in self.feature_cols:
            if c not in X_raw.columns:
                X_raw[c] = np.nan
        X_raw = X_raw[self.feature_cols]
        X_p   = self.preprocessor.transform(X_raw)
        X_vt  = self.vt.transform(X_p)
        return X_vt[:, self.sel_idx].astype(np.float32)

    def predict(self, X_raw):
        return self.model.predict(self._prep(X_raw))

    def predict_proba(self, X_raw):
        return self.model.predict_proba(self._prep(X_raw))

RISK_THRESHOLDS = {'LOW': (0.0, 0.35), 'MEDIUM': (0.35, 0.65), 'HIGH': (0.65, 1.01)}

def get_risk_level(prob, thresholds=RISK_THRESHOLDS):
    for level, (lo, hi) in thresholds.items():
        if lo <= prob < hi:
            return level
    return 'HIGH'

def predict_vehicle_status(input_data, pipeline=None, model_path='best_model.pkl'):
    import joblib
    if pipeline is None:
        pipeline = joblib.load(model_path)
    proba  = pipeline.predict_proba(input_data)[0]
    fail_p = float(proba[1])
    return {
        'prediction'          : 'Failure' if int(np.argmax(proba)) == 1 else 'Normal',
        'failure_probability' : round(fail_p * 100, 2),
        'risk_level'          : get_risk_level(fail_p),
        'status_ar'           : '🚨 عطل محتمل - ينصح بالفحص' if fail_p >= 0.5 else '✅ حالة ممتازة وآمنة',
    }
