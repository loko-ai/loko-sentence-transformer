from pathlib import Path
from typing import List, Optional

import joblib
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss


from setfit import SetFitModel, SetFitTrainer

from config.AppConfig import REPO_PATH
from utils.logger_utils import logger


class SentenceTransformer:
    def __init__(self, model_name: str = None, pretrained_name:str = None, is_multilabel: bool = False, multi_target_strategy: Optional[str] = None):
        self.model_name = model_name
        self.model = self._set_model(pretrained_name, is_multilabel, multi_target_strategy) if pretrained_name else pretrained_name

    def _set_model(self, pretrained_name, is_multilabel, multi_target_strategy):

        if is_multilabel:
            multi_target_strategy = multi_target_strategy or "one-vs-rest"
        model = SetFitModel.from_pretrained(pretrained_name, multi_target_strategy=multi_target_strategy)
        return model

    def fit(self, train_dataset, eval_dataset, loss, metric="accuracy", batch_size=16, n_iter=20, n_epochs=1, column_mapping=None):
        logger.debug("start fitting")

        if column_mapping is None:
            column_mapping={"sentence": "text", "label": "label"}
        self.trainer = SetFitTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss_class=loss,
            metric=metric,
            batch_size=batch_size,
            num_iterations=n_iter,  # The number of text pairs to generate for contrastive learning
            num_epochs=n_epochs,  # The number of epochs to use for constrastive learning
            column_mapping={"sentence": "text", "label": "label"}
            # Map dataset columns to text/label expected by trainer
        )
        logger.debug("trainer defined...")
        self.trainer.train()
        logger.debug("training")
        return "model fitted"

    def predict(self, text: List[str]):
        logger.debug("start prediction")
        preds = self.model(text)
        logger.debug(f"preds:::: {preds}")
        return preds

    def evaluate(self):
        logger.debug("start evaluate")
        metrics = self.trainer.evaluate()
        logger.debug(f"metrics::: {metrics}")
        return metrics

    def save(self):

        save_path = REPO_PATH / self.model_name
        save_path.mkdir(exist_ok=True, parents=True)
        joblib.dump(self.model, save_path/ "model")
        joblib.dump(self.trainer, save_path / "trainer")
        return "model saved"

    def load(self):
        load_path = REPO_PATH/ self.model_name
        self.model = joblib.load(load_path / "model")
        self.trainer = joblib.load(load_path / "trainer")
        return "model_loaded"




if __name__ == '__main__':
    dataset = load_dataset("sst2")
    print("dataset loaded")
    # Simulate the few-shot regime by sampling 8 examples per class
    num_classes = 2
    train_dataset = dataset["train"].shuffle(seed=42).select(range(8 * num_classes))
    eval_dataset = dataset["validation"]



    setfit = SentenceTransformer("example1","sentence-transformers/paraphrase-mpnet-base-v2")
    print("------------------------------------------------")
    setfit.fit(train_dataset, eval_dataset, CosineSimilarityLoss)

    print('------------------------------------------------')
    eval = setfit.evaluate()
    print(f"eval:::::: {eval}")
    print('------------------------------------------------')
    pred = setfit.predict(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
    print(pred)
    print('------------------------------------------------')
    res = setfit.save()
    print(res)

    print("model savedddddddddddddddddddddddddddddddddddddddd")
    st = SentenceTransformer(model_name="example1")
    res = st.load()
    print(res)
    print("model loadedddddddddddddddddddd ")
    print("------------------------------------------")
    eval = st.evaluate()
    print(f"eval:::::: {eval}")
    print("------------------------------------------")

    st.predict(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])

    print(pred)
    print('------------------------------------------------')