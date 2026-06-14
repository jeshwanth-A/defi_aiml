import torch
import numpy as np
from .model import model_load
from .data import fetch_model_window

def create_sequences(data, lookback):
    X = []

    for i in range(len(data[0]) - lookback):
        input_seq = []
        for j in range(i, i + lookback):
            input_seq.append([data[0][j],data[1][j],data[2][j]])
        X.append(input_seq)

    return np.array(X, dtype=np.float32)

async def predict_data() ->str:

    checkpoint = torch.load("ml/ethereum_lstm.pt", map_location="cpu")

    token = checkpoint["token"]
    days = checkpoint["days"]
    lookback = checkpoint["lookback"]

    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]

    feature_min = checkpoint["feature_min"]
    feature_max = checkpoint["feature_max"]

    prices, market_caps, volumes = await fetch_model_window(token, days)

    prices_np = np.array(prices, dtype=np.float32)
    market_caps_np = np.array(market_caps, dtype=np.float32)
    volumes_np = np.array(volumes, dtype=np.float32)


    normalized_price = (prices_np - feature_min[0]) / (feature_max[0] - feature_min[0])
    normalized_market_caps = (market_caps_np - feature_min[1]) / (feature_max[1] - feature_min[1])
    normalized_volumes = (volumes_np - feature_min[2]) / (feature_max[2] - feature_min[2])

    features = [
        normalized_price,
        normalized_market_caps,
        normalized_volumes
    ]

    X = create_sequences(features, lookback)
    X_tensor = torch.tensor(X, dtype=torch.float32).reshape(len(X), lookback, 3)

    model = model_load(input_size, hidden_size, num_layers)

    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)

    model.eval()

    with torch.no_grad():
        prediction = model(X_tensor[-1:])

    prediction_norm = prediction.cpu().numpy()

    prediction_norm = prediction_norm * (feature_max[0] - feature_min[0]) + feature_min[0]

    return f"predicted is {prediction_norm[-1][0]:.2f}, "
