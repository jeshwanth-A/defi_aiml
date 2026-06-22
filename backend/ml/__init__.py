from .data import fetch_exact_date_data, fetch_price_history, fetch_model_window

async def predict_data() -> str:
    from .predict import predict_data as run_prediction
    return await run_prediction()
