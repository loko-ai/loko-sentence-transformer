import importlib
import time
import json, tempfile
from pathlib import Path
from typing import List
from sanic.exceptions import SanicException

import datasets
from datasets import Dataset

from business.sentence_transformer_model import SentenceTransformer, LOSS_PATH
from config.AppConfig import REPO_PATH
from dao.fs_dao import FileSystemDAO
from model.model_bp import ModelInfo
from utils.logger_utils import logger
from utils.serialization_utils import serialize, deserialize


def load_params(params):
    res = dict()
    for k in params:
        v = params.get(k)
        if v == 'none':
            res[k] = None
        else:
            try:
                res[k] = json.loads(params.get(k))
            except:
                res[k] = params.get(k)
    return res


def get_all() -> List[str]:
    path = REPO_PATH
    fsdao = FileSystemDAO(path)
    return sorted(fsdao.all(files=False))

def check_existence(path:Path):
    if path.exists():
        return True
    else:
        return False

def create_st_model(model_name, pretrained_name, is_multilabel, multi_target_strategy, description=""):
    model_info = ModelInfo(model_name=model_name, pretrained_name=pretrained_name, is_multilabel=is_multilabel,
                           multi_target_strategy=multi_target_strategy, created_on=time.time(), description=description, fitted=False)
    model_info_dict = model_info.to_dict()
    logger.debug(f"blueprint data ----> {model_info_dict}")

    # model_info = dict(model_name=model_name, pretrained_name=pretrained_name, is_multilabel=is_multilabel,
    #                   multi_target_strategy=multi_target_strategy, fitt)
    model_path = REPO_PATH / model_name
    model_path.mkdir(exist_ok=True, parents=True)
    serialize(path=model_path, obj=model_info_dict)
    return f"Model '{model_name}' saved"


def get_datasets_format_data(data):
    dataset = Dataset.from_dict(data)
    # tfile = tempfile.NamedTemporaryFile(mode="w+", suffix=".json")
    # json.dump(data, tfile)
    # tfile.flush()
    # t_name = tfile.name.split("/")[-1]
    # t_path = "/".join(tfile.name.split("/")[:-1])
    # dataset = datasets.load_dataset(path=t_path, data_files=t_name)
    # tfile.close()
    return dataset


def fit_model_service(model_name, train_dataset, eval_dataset, fit_params, compute_eval_metrics):
    model_path = REPO_PATH / model_name
    try:
        model_blueprint = deserialize(path=model_path)

    except FileNotFoundError as notfound:
        logger.error("!!!!!! model corrupt !!!!!!")
        raise SanicException(f"Mredictor '{model_name}' doesn't exists or is corrupt...", status_code=404)
    model_blueprint["fitted"] = "Fitting"
    serialize(model_path, model_blueprint)

    model_info = ModelInfo(**model_blueprint)

    st = SentenceTransformer(model_name=model_name, pretrained_name=model_info.pretrained_name)
    # logger.debug(f"{res}")
    module = importlib.import_module(LOSS_PATH +fit_params["loss"])
    loss_kl = getattr(module, fit_params["loss"])
    st.fit(train_dataset=train_dataset, eval_dataset=eval_dataset, loss=loss_kl,
           metric=fit_params["metric"], n_epochs=fit_params["n_epochs"], n_iter=fit_params["n_iter"],
           batch_size=fit_params["batch_size"], column_mapping=fit_params["column_mapping"])
    logger.debug("model fitted")
    res = st.save()
    logger.debug(res)
    model_info.fitted = True
    model_info.fitting_date = time.time()
    logger.debug(f"fittig_date::: {model_info.fitting_date}")
    updated_blueprint = model_info.to_dict()
    logger.debug("updating blueprint")
    serialize(model_path, updated_blueprint)
    logger.debug("updated....")
    if compute_eval_metrics:
        metrics = st.evaluate()
        logger.debug(f"metrics::: {metrics}")
        return metrics
    else:
        res = f"model'{model_name}' fitted"
        logger.debug(f"model'{model_name}' fitted")
        return res


def eval_model_service(model_name, eval_dataset):
    st = SentenceTransformer(model_name=model_name, pretrained_name=None)
    res = st.load()
    logger.debug(res)
    metrics = st.evaluate(eval_dataset)
    logger.debug(f"metrics::: {metrics}")
    return metrics


def predict_model_service(model_name, test_dataset):
    st = SentenceTransformer(model_name=model_name, pretrained_name=None)
    res = st.load()
    logger.debug(res)
    preds = st.predict(test_dataset)
    preds_res = [dict(text=text_data, pred=str(pred)) for text_data, pred in zip(test_dataset, preds)]
    logger.debug(f"pred res:::: {preds_res}")
    return preds_res
