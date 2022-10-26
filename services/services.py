import re

from sanic import Sanic, Blueprint
from sanic_openapi import swagger_blueprint
from sanic_openapi.openapi2 import doc
from sanic_cors import CORS
from urllib.parse import unquote
from pathlib import Path
from sanic.response import raw
import sanic
from sanic.exceptions import SanicException, NotFound

from business.sentence_transformer import create_st_model
from config.AppConfig import REPO_PATH
from utils.logger_utils import logger
from utils.serialization_utils import serialize
from utils.service_utils import load_params

from loko_extensions.business.decorators import extract_value_args


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
@doc.tag('Loko Models')
@doc.summary("Save an object in 'models'")
@doc.description('''
    Examples
    --------

           ''')
@doc.consumes(doc.JsonBody({}), location="body")
@extract_value_args(file=False)
async def loko_create_model(value, args):
    model_name = args.get("model_name", None)
    if not model_name:
        raise ValueError("Model name must be specified...")
    # pretrained_name = args.get("pretrained_name", None)
    # is_multilabel = args.get("is_multilabel", false)
    # multi_target_strategy = args.get("multi_target_strategy", None)
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


#
#
# @bp.get("/models/<name>")
# @doc.tag('models')
# @doc.summary("Display object info from 'models'")
# @doc.consumes(doc.String(name="name"), location="path", required=True)
# async def models_info(request, name):
#     name = unquote(name)
#
#     path = repo_path / 'models' / name
#     if not path.exists():
#         raise SanicException(f"Model '{name}' does not exist!", status_code=400)
#     return sanic.json(deserialize(path))
#
#
# @bp.delete("/models/<name>")
# @doc.tag('models')
# @doc.summary("Delete an object from 'models'")
# @doc.consumes(doc.String(name="name"), location="path", required=True)
# async def delete_model(request, name):
#     name = unquote(name)
#
#     path = repo_path / 'models' / name
#     if not path.exists():
#         raise SanicException(f"Model '{name}' does not exist!", status_code=400)
#     shutil.rmtree(path)
#     return sanic.json(f'Model "{name}" deleted')
#
#
# @bp.post("/predictors/<name>/fit")
# @doc.tag('predictors')
# @doc.summary('Fit an existing predictor')
# @doc.description('''
#     Examples
#     --------
#
#                ''')
# # @doc.consumes(doc.String(name="fit_params"), location="query")
# @doc.consumes(doc.JsonBody({'data': doc.List(doc.Dictionary), 'target': doc.List()}), location="body", required=True)
# # @doc.consumes(doc.Integer(name="cv"), location="query")
# # @doc.consumes(doc.Boolean(name="partial"), location="query")
# @doc.consumes(doc.Float(name="test_size"), location="query")
# @doc.consumes(doc.Boolean(name="report"), location="query")
# @doc.consumes(doc.String(name="task", choices=['forecasting']), location="query",
#               required=True)  # 'classification', 'none'
# @doc.consumes(doc.Integer(name="forecasting_horizon"), location="query", required=False)
# @doc.consumes(doc.String(name="datetime_feature"), location="query", required=True)
# # @doc.consumes(doc.String(name="datetime_frequency", choices=["Years", "Months", "Days", "hours", "minutes", "seconds"]),
# #               required=True)
# @doc.consumes(doc.String(name="datetime_frequency"),
#               required=True)
# @doc.consumes(doc.String(name="name"), location="path", required=True)
# async def fit(request, name):
#     predictor_name = unquote(name)
#     fit_params = request.args
#     data = request.json
#
#     return sanic.json(f"Predictor '{predictor_name}' correctly fitted")
#
#
#
# @bp.post("/predictors/<name>/predict")
# @doc.tag('predictors')
# @doc.summary('Use an existing predictor to predict data')
# @doc.consumes(doc.JsonBody({'data': doc.List(doc.Dictionary)}), location="body", required=False)
# @doc.consumes(doc.Integer(name="forecasting_horizon"), location="query", required=False)
# @doc.consumes(doc.String(name="name"), location="path", required=True)
# async def predict(request, name):
#     predictor_name = unquote(name)
#     branch = "development"  # todo: aggiungere a fit e predict parametro branch
#     predict_params = {**load_params(request.args)}
#     data = request.json
#     preds = get_prediction(predictor_name=predictor_name, predict_params=predict_params, branch=branch, data=data)
#     return sanic.json(preds)
#
#
# @bp.post("/predictors/<name>/evaluate")
# @doc.tag('predictors')
# @doc.summary('Evaluate existing predictors in history')
# @doc.consumes(doc.JsonBody({'data': doc.List(doc.Dictionary), 'target': doc.List()}), location="body")
# # @doc.consumes(doc.JsonBody({}), location="body", required=False)
# # @doc.consumes(doc.Integer(name="forecasting_horizon"), location="query", required=False)
# @doc.consumes(doc.String(name="name"), location="path", required=True)
# async def evaluate(request, name):
#     branch = "development"
#     # params = {**load_params(request.args)}
#     params = dict()
#     params["branch"] = branch
#     name = unquote(name)
#     body = request.json
#
#     branch = "development"
#     # params = {**load_params(request.args)}
#     eval_params = dict(save_eval_report=False, eval_fname="NONE")
#
#     eval_res = get_model_evaluation(predictor_name=name, branch=branch, evaluate_params=eval_params,
#                                     data=body)
#     # logger.debug("loading predictor pipeline...")
#     # pipeline = load_pipeline(name, params['branch'], repo_path=repo_path)
#     # datetime = pipeline.date.strftime('%Y-%m-%d %H:%M:%S.%f')
#     #
#     # logger.debug("pre-processing evaluation data...")
#     # # y = FACTORY(body['target'])
#     # try:
#     #     data = preprocessing_data(body, datetime_feature=pipeline.datetime_feature,
#     #                               datetime_frequency=pipeline.datetime_frequency)
#     # except Exception as e:
#     #     print(f'----------------------{e}')
#     # y = data["y"]
#     # X = data["X"]
#     #
#     # logger.debug("computing forecast report")
#     # report = pipeline.get_forecast_report(y=y, X=X)
#     # logger.debug(f"report: {report}")
#     # res = [{"report_test": report, "datetime": datetime,
#     #         "task": "forecast"}]
#     # print("res:::", res)
#     return sanic.json(eval_res)
#
#
#
# @bp.post("/loko-services/predictors/fit")
# @doc.tag('loko-services')
# @doc.summary("...")
# @doc.consumes(doc.JsonBody({}), location="body")
# @extract_value_args(file=False)
# async def loko_fit_service(value, args):
#     logger.debug(f"fit::: value: {value}  \n \n args: {args}")
#     predictor_name = args["predictor_name"]
#     logger.debug(f"pred: {predictor_name}")
#     fit_params = FitServiceArgs(**args)
#
#     if not (fit_params.datetime_frequency and fit_params.datetime_feature):
#         msg = f"Date-time frequency value is '{fit_params.datetime_frequency}', Date-Time feature value is '{fit_params.datetime_feature}'. Both values need to be specified..."
#         logger.error(msg)
#         raise SanicException(msg, status_code=400)
#     logger.debug("-------------------------------------")
#     try:
#         train_model(predictor_name, fit_params=fit_params.to_dict(), data=value)
#     except Exception as e:
#         logger.error(f"Fitting LOG Error... {e}")
#         raise SanicException(f"Fitting LOG Error... {e}", status_code=500)
#     res = f"Predictor '{predictor_name}' correctly fitted"
#     # res = get_all('transformers')
#     # save_defaults(repo='transformers')
#     return sanic.json(res)
#
#
# @bp.post("/loko-services/predictors/predict")
# @doc.tag('loko-services')
# @doc.summary("Use an existing predictor to predict data")
# @extract_value_args(file=False)
# async def loko_predict_service(value, args):
#     logger.debug(f"predict::: value: {value}  \n \n args: {args}")
#     if isinstance(value, str):
#         value = None
#     branch = "development"
#     predictor_name = args["predictor_name"]
#
#     logger.debug(f"pred: {predictor_name}")
#     predict_params = PredictServiceArgs(**args).to_dict()
#     logger.debug(f"predict_params: {predict_params}")
#     # print("si")
#     # res = get_all('transformers')
#     # save_defaults(repo='transformers')
#     try:
#         res = get_prediction(predictor_name, predict_params, branch=branch, data=value)
#     except Exception as e:
#         logger.error(f"Prediction LOG err {e}")
#         raise e
#     return sanic.json(res)
#
#
# @bp.post("/loko-services/predictors/evaluate")
# @doc.tag('loko-services')
# @doc.summary("...")
# @doc.consumes(doc.JsonBody({}), location="body")
# @extract_value_args(file=False)
# async def loko_evaluate_service(value, args):
#     logger.debug(f"evaluate::: value: {value}  \n \n args: {args}")
#
#
#     branch = "development"
#     # params = {**load_params(request.args)}
#     predictor_name = unquote(args["predictor_name"])
#     logger.debug(f"pred: {predictor_name}")
#     eval_params = EvaluateServiceArgs(**args).to_dict()
#
#     try:
#         eval_res = get_model_evaluation(predictor_name=predictor_name, branch=branch, evaluate_params=eval_params,
#                                         data=value)
#     except Exception as e:
#         logger.error(f"Evaluate LOG err: {e}")
#         raise SanicException(f"Evaluate LOG Error... {e}")
#     return sanic.json(eval_res)
#
#
#
#
# @bp.post("/loko-services/info_obj")
# @doc.tag('loko-services')
# @doc.summary("Get info from 'models', 'transformer' or 'predictor' - service compatible with loko;")
# @doc.consumes(doc.JsonBody({}), location="body")
# @extract_value_args(file=False)
# async def loko_info_service(value, args):
#     logger.debug(f"infooo::: value: {value}  \n \n args: {args}")
#     info_obj = args.get("info_obj", None)
#     logger.debug(f"info obj {info_obj}")
#     obj_name = args.get("info_obj_name", None)
#     if info_obj is None:
#         msg = "Object to get info on not specified! Please select one of the option..."
#         logger.error(msg)
#         raise SanicException(msg, status_code=400)
#     if not obj_name:
#         msg = "Object name to get info on not specified! Please select one of them from the available list..."
#         logger.error(msg)
#         raise SanicException(msg, status_code=400)
#     obj = info_obj.lower() + "s"
#     path = repo_path / obj / obj_name
#
#     logger.debug(f"debugging url {path}")
#     infos = deserialize(path)
#     logger.debug(f"res:::::{infos}")
#
#     return sanic.json(infos)

app.blueprint(bp)
if __name__ == "__main__":
    app.run("0.0.0.0", 8080)
