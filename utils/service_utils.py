import json


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
