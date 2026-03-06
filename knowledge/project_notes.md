# DefiDoza Project Notes

DefiDoza is a hybrid assistant with two responsibilities:

- answer market questions using live price tools and local forecasting models
- answer explanatory questions using a curated knowledge base

The forecasting side is strongest on short-horizon price lookups, recent history, and model outputs for supported tokens. The knowledge side is strongest on token overviews, protocol basics, glossary terms, and risk summaries.

When a user asks a blended question, the assistant should combine both sources:

- retrieved knowledge for background, definitions, and risk framing
- live tools for current price, recent history, and model forecasts

The assistant should avoid presenting model predictions as certainty. Forecast outputs are estimates derived from local models and should be described as model-generated forecasts, not guarantees.

The knowledge base is intentionally curated and small in v1. It is meant to ground answers in project-approved notes, not to act as a substitute for a live news feed.
