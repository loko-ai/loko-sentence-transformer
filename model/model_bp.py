

class ModelInfo():
    def __init__(self, model_name, pretrained_name, is_multilabel, multi_target_strategy, fitted=False, fitting_date=None):
        self.model_name = model_name
        self.pretrained_name = pretrained_name
        self.is_multilabel = is_multilabel
        self.multi_target_strategy = multi_target_strategy
        self.fitted = fitted
        self.fitting_date = None

    def to_dict(self):
        return self.__dict__