import os
import pickle

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor

from core.config import (
    PYTORCH_WEIGHTS,
    TF_WEIGHTS,
    RF_WEIGHTS,
    admin_log,
)


class PricePredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_tf_model(shape):
    m = Sequential([LSTM(50, input_shape=shape), Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    return m


def create_rf_model():
    return RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)


def weights_exist(name):
    return os.path.exists(
        {
            "pytorch": PYTORCH_WEIGHTS,
            "tensorflow": TF_WEIGHTS,
            "randomforest": RF_WEIGHTS,
        }.get(name, "")
    )


def get_trained_models():
    return [m for m in ["pytorch", "tensorflow", "randomforest"] if weights_exist(m)]


def load_model(name, input_size=5):
    admin_log("MODEL", f"Loading {name}...")
    if name == "pytorch":
        m = PricePredictor(input_size=input_size)
        m.load_state_dict(torch.load(PYTORCH_WEIGHTS, map_location="cpu"))
        m.eval()
        admin_log("MODEL", f"Loaded {name} (input_size={input_size})")
        return m
    elif name == "tensorflow":
        m = tf.keras.models.load_model(TF_WEIGHTS, compile=False)
        admin_log("MODEL", f"Loaded {name}")
        return m
    elif name == "randomforest":
        with open(RF_WEIGHTS, "rb") as f:
            m = pickle.load(f)
        admin_log("MODEL", f"Loaded {name}")
        return m
    return None
