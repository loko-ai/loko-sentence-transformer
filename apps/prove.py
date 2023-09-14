from sentence_transformers import losses
import joblib as joblib


# print(losses)

model = joblib.load("example_model")
preds = model(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
print(f"preds:: {preds}")
