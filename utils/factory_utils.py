import importlib
import sys


def get_factory(obj, klass="__klass__"):
    if isinstance(obj, dict):
        if klass in obj:
            kl_path = obj[klass]
            mod_path = ".".join(kl_path.split(".")[:-1])
            kl_name = kl_path.split(".")[-1]
            module = importlib.import_module(mod_path)
            kl = getattr(module, kl_name)
            args = get_factory({k: get_factory(v) for k, v in obj.items() if k != klass})
            return kl(**args)
        else:
            for k, v in obj.items():
                return {k: get_factory(v) for (k, v) in obj.items()}
    if isinstance(obj, list):
        return [get_factory(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(get_factory(v) for v in obj)
    return obj



if __name__ == '__main__':
    loss = "CosineSimilarityLoss"
    loss_path = "sentence_transformers.losses."
    module = importlib.import_module(loss_path+loss)
    print(module)
    kl = getattr(module, loss)
    print(kl)
    # es = dict(__klass__=loss_path+loss)
    # res = get_factory(es)
    # getattr(sys.modules[__name__], str)