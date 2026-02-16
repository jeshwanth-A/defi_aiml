import os
import pickle
import warnings

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover
    InconsistentVersionWarning = Warning

from core.config import (
    PYTORCH_WEIGHTS,
    TF_WEIGHTS,
    RF_WEIGHTS,
    admin_log,
)


class PricePredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_tf_model(shape):
    m = Sequential([LSTM(64, input_shape=shape), Dropout(0.2), Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    return m


def create_rf_model():
    return RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )


def _tf_weight_candidates():
    candidates = [TF_WEIGHTS]
    root, ext = os.path.splitext(TF_WEIGHTS)
    if ext == ".h5":
        candidates.append(f"{root}.keras")
    elif ext == ".keras":
        candidates.append(f"{root}.h5")

    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def _resolve_tf_weights_path():
    for path in _tf_weight_candidates():
        if os.path.exists(path):
            return path
    return None


def _infer_pytorch_shape(state_dict, fallback_input_size):
    key = "lstm.weight_ih_l0"
    if key not in state_dict:
        return fallback_input_size, 64, 2

    first = state_dict[key]
    hidden_size = int(first.shape[0] // 4)
    input_size = int(first.shape[1])
    num_layers = 0
    while f"lstm.weight_ih_l{num_layers}" in state_dict:
        num_layers += 1

    return input_size, hidden_size, max(1, num_layers)


def _load_pickle_weights(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        with open(path, "rb") as f:
            return pickle.load(f)


def weights_exist(name):
    if name == "tensorflow":
        return _resolve_tf_weights_path() is not None
    return os.path.exists(
        {
            "pytorch": PYTORCH_WEIGHTS,
            "randomforest": RF_WEIGHTS,
        }.get(name, "")
    )


def get_trained_models():
    return [m for m in ["pytorch", "tensorflow", "randomforest"] if weights_exist(m)]


def load_model(name, input_size=5):
    admin_log("MODEL", f"Loading {name}...")
    if name == "pytorch":
        if not os.path.exists(PYTORCH_WEIGHTS):
            admin_log("MODEL", f"Missing weights: {PYTORCH_WEIGHTS}")
            return None
        try:
            state = torch.load(PYTORCH_WEIGHTS, map_location="cpu")
            resolved_input, hidden_size, num_layers = _infer_pytorch_shape(
                state, input_size
            )
            m = PricePredictor(
                input_size=resolved_input,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
            m.load_state_dict(state)
            m.eval()
            admin_log(
                "MODEL",
                f"Loaded {name} (input_size={resolved_input}, hidden={hidden_size}, layers={num_layers})",
            )
            return m
        except Exception as e:
            admin_log("MODEL", f"Failed loading {name}: {str(e)[:180]}")
            return None
    elif name == "tensorflow":
        tf_path = _resolve_tf_weights_path()
        if not tf_path:
            admin_log("MODEL", "Missing TensorFlow weights (.h5/.keras)")
            return None
        try:
            m = tf.keras.models.load_model(tf_path, compile=False)
            admin_log("MODEL", f"Loaded {name} ({os.path.basename(tf_path)})")
            return m
        except Exception as e:
            admin_log("MODEL", f"Failed loading {name}: {str(e)[:180]}")
            return None
    elif name == "randomforest":
        if not os.path.exists(RF_WEIGHTS):
            admin_log("MODEL", f"Missing weights: {RF_WEIGHTS}")
            return None
        try:
            m = _load_pickle_weights(RF_WEIGHTS)
            admin_log("MODEL", f"Loaded {name}")
            return m
        except Exception as e:
            admin_log("MODEL", f"Failed loading {name}: {str(e)[:180]}")
            return None
    return None
