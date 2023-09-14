import json

import joblib as joblib
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("sst2")
print("dataset loaded")
# Simulate the few-shot regime by sampling 8 examples per class
num_classes = 2
train_dataset = dataset["train"].shuffle(seed=42).select(range(8 * num_classes))
eval_dataset = dataset["validation"]

print("preprocessing done")
print(train_dataset.data)

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

print("model selectes")
# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=16,
    num_iterations=20,  # The number of text pairs to generate for contrastive learning
    num_epochs=1,  # The number of epochs to use for constrastive learning
    column_mapping={"sentence": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
)

print("model")
# Train and evaluate
trainer.train()

# print(model.__dict__)
joblib.dump(model, "example_model")


metrics = trainer.evaluate()
print(f"metrics::: {metrics}")


# Push model to the Hub
# trainer.push_to_hub("my-awesome-setfit-model")
#ValueError: You must login to the Hugging Face hub on this computer by typing `huggingface-cli login` and entering your credentials to use `use_auth_token=True`. Alternatively, you can pass your own token as the `use_auth_token` argument.

# Download from Hub and run inference
# model = SetFitModel.from_pretrained("lewtun/my-awesome-setfit-model")
# Run inference
preds = model(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
print(f"preds:: {preds}")
body = [dict(pred=list(preds))]
print(body)
es = json.dumps([body])
