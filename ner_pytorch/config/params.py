import json
import yaml
from typing import Dict, List, Union

from dotenv import dotenv_values


class Params:
    """
    C'est un peu pénible d'accéder aux paramètres d'un dictionnaire
    avec une syntaxe comme PARAMS['MODELS']['TF_IDF']['NGRAM_RANGE']...
    La classe Params simplifie cette syntaxe et permet en outre
    l'autocomplétion depuis les notebooks !

    Remarques :
    - on en profite pour passer les clés en uppercase, comme c'est la
    norme pour les constantes en Python,
    - cette classe est récursive.

    Usage :
    >>> PARAMS = Params(params_path='params.yml')
    >>> PARAMS.EMPTY_PARAMS
    True
    """

    def __init__(self, params_path: str = None,
                 params: Union[Dict, List] = None,
                 key: str = None):
        if params_path is not None:
            with open(params_path) as file:
                self.params = yaml.safe_load(file)
            self._gfk(self.params, key='start')
        else:
            self.params = params
            self._gfk(js=params, key=key)

    def _gfk(self, js, key):
        if not isinstance(js, dict):
            setattr(self, key.upper(), js)
        else:
            for k, v in js.items():
                if isinstance(v, dict):
                    setattr(self, k.upper(), Params(params=v, key=k))
                else:
                    setattr(self, k.upper(), js[k])

    def get(self, item):
        return self.params.get(item)

    def update(self, dico: dict):
        self.params.update(dico)

    def copy(self):
        return self.params.copy()

    def setdefault(self, itemA, itemB):
        if itemA in self.params:
            return self.params[itemA]
        self.params[itemA] = itemB
        return itemB
    
    def __repr__(self):
        return json.dumps(self.params)

    def __iter__(self):
        for k in self.params:
            yield k

    def __getitem__(self, item):
        return self.params.get(item)

# pwd --> 'NER PyTorch/notebooks'
PARAMS = Params(params_path='ner_pytorch/config/params.yml')
PARAMS.update(dotenv_values('ner_pytorch/config/.env'))
