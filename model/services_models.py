

class FitServiceArgs():
    def __init__(self, loss, metric, batch_size, n_iter, n_epochs, text_feature, label_feature, learning_rate, **kwargs):
        self.label_feature = label_feature
        self.text_feature = text_feature
        self.n_epochs = n_epochs
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.metric = metric
        self.loss = loss
        self.learning_rate = learning_rate
        self.column_mapping = self._create_column_mapping()


    def _create_column_mapping(self):
        column_mapping = {self.text_feature:"text", self.label_feature:"label"}
        return column_mapping

    def to_dict(self):
        return self.__dict__