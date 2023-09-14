import importlib
import tempfile, json

from datasets.arrow_dataset import Dataset
from datasets import Dataset
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


dataset = Dataset.from_dict(example)
print(dataset.data)