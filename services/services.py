import re

import requests

from sanic import Sanic, Blueprint
from sanic_openapi import swagger_blueprint
from sanic_openapi.openapi2 import doc
from sanic_cors import CORS
from urllib.parse import unquote
import sanic
from sanic.exceptions import SanicException

from model.services_models import FitServiceArgs
from utils.services_utils import create_st_model, fit_model_service, eval_model_service, get_datasets_format_data, \
    predict_model_service
from config.AppConfig import REPO_PATH, ORCHESTRATOR
from utils.logger_utils import logger
from utils.services_utils import load_params

from loko_extensions.business.decorators import extract_value_args

import datasets


def get_app(name):
    app = Sanic(name)
    swagger_blueprint.url_prefix = "/api"
    app.blueprint(swagger_blueprint)
    return app


repo_path = REPO_PATH
name = "sentence_transformer"
app = get_app(name)
# url_prefix=f"ds4biz/time_series/{get_pom_major_minor()}")
bp = Blueprint("default")
app.config["API_TITLE"] = name
# app.config["REQUEST_MAX_SIZE"] = 20000000000 ## POI TOGLIERE!!
CORS(app)


# app.static("/web", "/frontend/dist")


### MODELS ###

# @bp.get("/models")
# @doc.tag('models')
# @doc.summary("List objects in 'models'")
# async def list_models(request):
#     # save_defaults(repo='models')
#     return sanic.json(get_all('models'))


@bp.get("/models/<model_name>")
@doc.tag('models')
@doc.summary("Save an object in 'models'")
@doc.description('''
    Examples
    --------
   
           ''')
@doc.consumes(doc.String(name="multi_target_strategy"), location="query", required=False)
@doc.consumes(doc.Boolean(name="is_multilabel"), location="query", required=False)
@doc.consumes(doc.String(name="pretrained_name"), location="query", required=True)
@doc.consumes(doc.String(name="model_name"), location="path", required=True)
async def create_model(request, model_name):
    model_name = unquote(model_name)
    default_params = dict(is_multilabel=False,
                          multi_target_strategy=None,
                          )
    model_params = {**default_params, **load_params(request.args)}
    print(f"params {model_params}")
    # pretrained_model = request.get("pretrained_name")
    # is_multilabel = request.get

    if not re.search(r'(?i)^[a-z0-9]([a-z0-9_]*[a-z0-9])?$', model_name):
        raise SanicException('No special characters (except _ in the middle of name) and whitespaces allowed',
                             status_code=400)

    try:
        create_st_model(model_name=model_name, **model_params)
    except Exception as e:
        raise e
    return sanic.json(f"Model '{model_name}' saved")


@bp.post("/model/create")
@doc.tag('Loko Model Services')
@doc.summary("Save an object in 'models'")
@doc.description('''
''')
@doc.consumes(doc.JsonBody({}), location="body")
@extract_value_args(file=False)
async def loko_create_model(value, args):
    model_name = args.get("model_name", None)
    if not model_name:
        raise ValueError("Model name must be specified...")

    default_params = dict(is_multilabel=False,
                          multi_target_strategy=None,
                          )
    params = {**default_params, **load_params(args)}

    logger.debug(f"model params:::::::::  {params}")

    if not re.search(r'(?i)^[a-z0-9]([a-z0-9_]*[a-z0-9])?$', model_name):
        raise SanicException('No special characters (except _ in the middle of name) and whitespaces allowed',
                             status_code=400)

    try:
        create_st_model(model_name=params["model_name"], pretrained_name=params["pretrained_name"],
                        is_multilabel=params["is_multilabel"], multi_target_strategy=params["multi_target_strategy"])
    except Exception as e:
        raise e
    return sanic.json(f"Model '{model_name}' saved")


@bp.post("/model/fit")
@doc.tag('Loko Model Services')
@doc.summary("Fit a sentence transformer object")
@doc.description('''
''')
@doc.consumes(doc.JsonBody({}), location="body")
@extract_value_args(file=False)
async def loko_fit_model(value, args):
    logger.debug(f'valueeeee:::::: {value}')
    logger.debug(f"argssss=========== {args}")
    data = value
    # if train_path:
    #     file_writer_path = ORCHESTRATOR + "files" + train_path
    #     logger.debug(f"GW url for writing file: {file_writer_path}")
    #     train_dataset = requests.get(file_writer_path)
    # else:
    #     raise

    if "eval_dataset" in data:
        eval_data = data.get("eval_dataset", None)
        eval_dataset = get_datasets_format_data(eval_data)
        train_data = data.get("train_dataset", None)
        train_dataset = get_datasets_format_data(train_data)
    else:
        train_data = data.copy()
        train_dataset = get_datasets_format_data(train_data)
        logger.debug(f"train!!!!! {train_dataset}")
        eval_dataset = None



    # train_dataset = value.get("train_dataset", None)
    compute_eval_metrics = eval(value.get("compute_eval_metrics", "false").capitalize())
    # eval_dataset = value.get("eval_dataset", None)

    model_name = args.get("model_name", None)
    if not model_name:
        raise ValueError("Model name must be specified...")

    default_params = dict(
        model=None,
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        loss=None,
        metric="accuracy",
        batch_size=16,
        num_iterations=10,  # The number of text pairs to generate for contrastive learning
        num_epochs=1,  # The number of epochs to use for constrastive learning
        learning_rate=0.000_02,
        column_mapping={"text": "text", "label": "label"}
    )
    fit_params = FitServiceArgs(**load_params(args)).to_dict()

    params = {**default_params, **fit_params}

    logger.debug(f"model params:::::::::  {params}")


    if not re.search(r'(?i)^[a-z0-9]([a-z0-9_]*[a-z0-9])?$', model_name):
        raise SanicException('No special characters (except _ in the middle of name) and whitespaces allowed',
                             status_code=400)

    try:
        res = fit_model_service(model_name, train_dataset, eval_dataset, params, compute_eval_metrics)
    except Exception as e:
        raise e

    return sanic.json(res)


@bp.post("/model/evaluate")
@doc.tag('Loko Model Services')
@doc.summary("Evaluate a sentence transformer object")
@doc.description('''
''')
@doc.consumes(doc.JsonBody({}), location="body")
@extract_value_args(file=False)
async def loko_evaluate_model(value, args):
    eval_data = value.copy()
    eval_dataset = get_datasets_format_data(eval_data)

    model_name = args.get("model_name", None)
    if not model_name:
        raise ValueError("Model name must be specified...")

    try:
        res = eval_model_service(model_name, eval_dataset)
    except Exception as e:
        raise e
    return sanic.json(res)

@bp.post("/model/predict")
@doc.tag('Loko Model Services')
@doc.summary("Predict a sentence transformer object")
@doc.description('''
''')
@doc.consumes(doc.JsonBody({}), location="body")
@extract_value_args(file=False)
async def loko_evaluate_model(value, args):
    test_data = value

    model_name = args.get("model_name", None)
    if not model_name:
        raise ValueError("Model name must be specified...")

    try:
        res = predict_model_service(model_name, test_data)
    except Exception as e:
        raise e
    return sanic.json(res)


app.blueprint(bp)
if __name__ == "__main__":
    app.run("0.0.0.0", 8080)
