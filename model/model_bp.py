

class ModelInfo():
    def __init__(self, model_name, pretrained_name, is_multilabel, multi_target_strategy, created_on=None, description="", fitted=False, fitting_date=None):
        self.model_name = model_name
        self.pretrained_name = pretrained_name
        self.created_on = created_on
        self.description = description
        self.is_multilabel = is_multilabel
        self.multi_target_strategy = multi_target_strategy
        self.fitted = fitted
        self.fitting_date = fitting_date

    def to_dict(self):
        return self.__dict__