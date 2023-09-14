import os
from pathlib import Path


class FileSystemDAO:

    def __init__(self, path, history=False, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)
        self.path = path
        # self.history = history

    def save(self):
        raise Exception("not implemented")

    def update(self):
        raise Exception("not implemented")

    def all(self, files=True):
        if files:
            lista = []
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    lista.append(os.path.join(root,file))
            # if self.history:
            #     return history_parse(lista)
            return lista
        if os.path.exists(self.path):

            return os.listdir(self.path)

        return []

    def get_by_id(self):
        raise Exception("not implemented")

    def remove(self):
        raise Exception("not implemented")