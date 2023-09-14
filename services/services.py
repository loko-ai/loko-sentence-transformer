import io
import re
import shutil
import traceback

import requests

from sanic import Sanic, Blueprint
from sanic_openapi import swagger_blueprint
from sanic_openapi.openapi2 import doc
from sanic_cors import CORS
from urllib.parse import unquote
import sanic
from sanic.exceptions import SanicException, NotFound

from model.services_models import FitServiceArgs
from utils.serialization_utils import deserialize
from utils.services_utils import create_st_model, fit_model_service, eval_model_service, get_datasets_format_data, \
    predict_model_service, get_all, check_existence
from config.AppConfig import REPO_PATH, ORCHESTRATOR
from utils.logger_utils import logger
from utils.services_utils import load_params

from loko_extensions.business.decorators import extract_value_args

import datasets

from utils.zip_utils import import_zipfile, make_zipfile


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


app.static("/web", "/frontend/dist")


### MODELS ###

@bp.get("/models")
@doc.tag('models')
@doc.summary("List objects in 'models'")
async def list_models(request):
    # save_defaults(repo='models')
    return sanic.json(get_all())


@bp.delete("/models/<model_name>")
@doc.tag('models')
@doc.summary("Delete a model object")
@doc.consumes(doc.String(name="model_name"), location="path", required=True)
async def delete_predictor(request, model_name):
    model_name = unquote(model_name)

    path = repo_path / model_name
    if not path.exists():
        raise SanicException(f'Model "{model_name}" does not exist!', status_code=400)
    # ##TODO: sviluppare questa parte
    # if name in fitting.all('alive'):
    #     dc = fitting.get_by_id(name)['dc']
    #     cd.kill(dc.name)
    #     msg = 'Killed'
    #     fitting.add(name, msg)
    #     send_message(name, msg)
    #     fitting.remove(name)
    #     logger.debug(f'Fitting {name} Killed')
    # else:
    #     cname = list(filter(lambda x: x.endswith('_'+name), cd.containers.keys()))
    #     if cname:
    #         cd.kill(cname[0])
    #         logger.debug(f'Container {cname[0]} Killed')

    shutil.rmtree(path)

    # testset_dao = get_dataset_dao(repo_path=repo_path)
    # testset_dao.set_coll(name)
    # testset_dao.dropcoll()
    # testset_dao.close()

    return sanic.json(f"Model '{model_name}' deleted")


@bp.post("/models/<model_name>")
@doc.tag('models')
@doc.summary("Save an object in 'models'")
@doc.description(''' ''') #todo: add example
@doc.consumes(doc.String(name="multi_target_strategy"), location="query", required=False)
@doc.consumes(doc.Boolean(name="is_multilabel"), location="query", required=False)
@doc.consumes(doc.String(name="pretrained_name"), location="query", required=True)
@doc.consumes(doc.String(name="description"), location="query", required=False)
@doc.consumes(doc.String(name="model_name"), location="path", required=True)
async def create_model(request, model_name):
    print(f"name::: {model_name}")

    model_name = unquote(model_name)
    default_params = dict(is_multilabel=False,
                          multi_target_strategy=None,
                          description=""
                          )
    model_params = {**default_params, **load_params(request.args)}
    # pretrained_model = request.get("pretrained_name")
    # is_multilabel = request.get

    if not re.search(r'(?i)^[a-z0-9]([a-z0-9_]*[a-z0-9])?$', model_name):
        raise SanicException('No special characters (except _ in the middle of name) and whitespaces allowed',
                             status_code=400)

    try:
        create_st_model(model_name=model_name, **model_params)
    except Exception as e:
        logger.error(f"CREATE MODEL ERROR: {e}")
        raise SanicException(e)
    return sanic.json(f"Model '{model_name}' saved")


@bp.get("/models/<name>")
@doc.tag('models')
@doc.summary("Display object info from 'models'")
@doc.consumes(doc.String(name="name"), location="path", required=True)
async def models_info(request, name):
    name = unquote(name)

    path = repo_path / name
    if not path.exists():
        raise SanicException(f"Model '{name}' does not exist!", status_code=400)
    return sanic.json(deserialize(path))


@bp.post("/models/import")
@doc.tag('models')
@doc.summary('Upload existing model')
@doc.consumes(doc.File(name="f"), location="formData", content_type="multipart/form-data", required=True)
async def import_model(request):
    print(f"repo::: {repo_path}")
    file = request.files.get('f')
    if file.name.endswith('.zip'):
        import_zipfile(file, repo_path)
    else:
        raise Exception("Importing Error...")
    return sanic.json('Model correctly imported')


@bp.get("/models/<model_name>/export")
@doc.tag('models')
@doc.summary('Download existing model')
@doc.consumes(doc.String(name="model_name"), location="path", required=True)
async def export_model(request, model_name):

    model_name = unquote(model_name)
    print(f"------ {model_name}")

    file_name = model_name + '.zip'
    path = repo_path / model_name
    print("qio")
    buffer = io.BytesIO()
    print("qiuii")
    make_zipfile(buffer, path)
    buffer.seek(0)
    headers = {'Content-Disposition': 'attachment; filename="{}"'.format(file_name)}
    return sanic.response.raw(buffer.getvalue(), headers=headers)


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
                          description=""
                          )
    params = {**default_params, **load_params(args)}

    logger.debug(f"model params:::::::::  {params}")

    if not re.search(r'(?i)^[a-z0-9]([a-z0-9_]*[a-z0-9])?$', model_name):
        raise SanicException('No special characters (except _ in the middle of name) and whitespaces allowed',
                             status_code=400)

    try:
        create_st_model(model_name=params["model_name"], pretrained_name=params["pretrained_name"],
                        is_multilabel=params["is_multilabel"], multi_target_strategy=params["multi_target_strategy"], description=params["description"])
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
    logger.debug(f'value type: {type(value)}')
    logger.debug(f"input args: {args}")
    data = value
    # if train_path:
    #     file_writer_path = ORCHESTRATOR + "files" + train_path
    #     logger.debug(f"GW url for writing file: {file_writer_path}")
    #     train_dataset = requests.get(file_writer_path)
    # else:
    #     raise

    model_name = args.get("model_name", None)
    model_path = REPO_PATH / model_name
    if not check_existence(model_path) or model_name == "":
        raise SanicException(f"Model '{model_name}' doesn't exists", status_code=404)

    if not model_name:
        raise ValueError("Model name must be specified...")

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
        compute_eval_metrics=False,
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


@app.exception(Exception)
async def manage_exception(request, exception):
    status_code = getattr(exception, "status_code", None) or 500
    logger.debug(f"status_code:::: {status_code}")
    if isinstance(exception, SanicException):
        return sanic.json(dict(error=str(exception)), status=status_code)

    e = dict(error=f"{exception.__class__.__name__}: {exception}")

    if isinstance(exception, NotFound):
        return sanic.json(e, status=404)
    # logger.error(f"status code {status_code}")
    logger.error('TracebackERROR: \n' + traceback.format_exc() + '\n\n', exc_info=True)
    return sanic.json(e, status=status_code)

app.blueprint(bp)
if __name__ == "__main__":
    app.run("0.0.0.0", 8080)
