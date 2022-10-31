import importlib
import tempfile, json

import datasets
from sentence_transformers.losses import CosineSimilarityLoss


from business.sentence_transformer_model import SentenceTransformer




example = {
  "idx": [
    32326,
    27449,
    60108,
    23141,
    35226,
    66852,
    65093,
    47847,
    39440,
    56428,
    6798,
    6824,
    56503,
    51288,
    39024,
    58847
  ],
  "sentence": [
    "klein , charming in comedies like american pie and dead-on in election , ",
    "be fruitful ",
    "soulful and ",
    "the proud warrior that still lingers in the souls of these characters ",
    "covered earlier and much better ",
    "wise and powerful ",
    "a powerful and reasonably fulfilling gestalt ",
    "smart and newfangled ",
    "it too is a bomb . ",
    "guilty about it ",
    "while the importance of being earnest offers opportunities for occasional smiles and chuckles ",
    "stevens ' vibrant creative instincts ",
    "great artistic significance ",
    "what does n't this film have that an impressionable kid could n't stand to hear ? ",
    "working from a surprisingly sensitive script co-written by gianni romoli ... ",
    "eight crazy nights is a total misfire . "
  ],
  "label": [
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    0
  ]
}
# a = datasets.load_dataset_builder((example))
# print(a)
############TODO: problema: i dati veri vengono visti come una lista di liste [[0,0,1,0,0,...,0,1,0,0,1]], mentre quelli caricati [[[1,1,1,1,0,...,1,1,1,1,0]]]
# example = {"text": ["ciaiaoaoa sjiewheiugwfedmas dsaadbb jjhew;ew", "sjcudsuussu"],
# "label":  [1, 2]}
tfile = tempfile.NamedTemporaryFile(mode="w+", suffix=".json")
json.dump(example, tfile)
tfile.flush()
print(tfile.name.split("/")[-1])
nome = tfile.name.split("/")[-1]
path = "/".join(tfile.name.split("/")[:-1])
train_dataset = datasets.load_dataset(path=path,data_files=nome, split="train")#["train"]
train_dataset = datasets.Dataset.from_dict(example)
print(train_dataset)
print("----------------------")
print(train_dataset.column_names)
setfit = SentenceTransformer("example1" ,"sentence-transformers/paraphrase-mpnet-base-v2")

print("------------------------------------------------")
loss = "CosineSimilarityLoss"
loss_path = "sentence_transformers.losses."
column_mapping = {"sentence": "text", "label": "label"}
module = importlib.import_module(loss_path +loss)
kl = getattr(module, loss)


setfit.fit(train_dataset,None, kl, column_mapping=column_mapping)
#
# print('------------------------------------------------')
# eval = setfit.evaluate()
# print(f"eval:::::: {eval}")
# print('------------------------------------------------')
# pred = setfit.predict(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
# print(pred)
# print('------------------------------------------------')
# res = setfit.save()
# print(res)
#
# print("model savedddddddddddddddddddddddddddddddddddddddd")
# st2 = SentenceTransformer(model_name="example1")
# res = st2.load()
# print(res)
# print("model loadedddddddddddddddddddd ")
# print("------------------------------------------")
# eval = st2.evaluate()
# print(f"eval:::::: {eval}")
# print("------------------------------------------")
#
# st2.predict(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
#
# print(pred)
# print('------------------------------------------------')