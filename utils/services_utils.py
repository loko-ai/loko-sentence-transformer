from config.AppConfig import REPO_PATH
from utils.serialization_utils import serialize


def create_st_model(model_name, pretrained_name, is_multilabel, multi_target_strategy):
    model_info = dict(model_name=model_name, pretrained_name=pretrained_name, is_multilabel=is_multilabel,
                      multi_target_strategy=multi_target_strategy)
    model_path = REPO_PATH / model_name
    model_path.mkdir(exist_ok=True, parents=True)
    serialize(path=model_path, obj=model_info)
    return f"Model '{model_name}' saved"




