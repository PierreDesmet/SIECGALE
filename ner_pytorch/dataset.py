import ast
import torch
from transformers import BertTokenizer, CamembertTokenizer

from ner_pytorch.config.params import PARAMS


class EntityDataset(torch.utils.data.Dataset):
    if PARAMS.LANGUAGE == 'en':
        tokenizer = BertTokenizer.from_pretrained(
            PARAMS.PATHS_EN.MODEL, 
            do_lower_case=PARAMS.MODEL.DO_LOWER_CASE
        )
    elif PARAMS.LANGUAGE == 'fr':
        tokenizer = CamembertTokenizer.from_pretrained(
            PARAMS.PATHS_FR.MODEL  #, proxies=PARAMS.PROXIES
        ) 
        
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags
        
        func = self.tokenizer.encode
        self.special_token_start, *_, self.special_token_end = func("Test", 
                                                                   add_special_tokens=True)
        self.PADDING_VALUE = self.tokenizer.pad_token_id

    def getitem_enrichi(self, item: int):
        print(self.texts[item])
        return self.__getitem__(item)
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, position: int):
        liste_mots: list = self.texts[position]
        tags = self.tags[position]
        assert len(tags) == len(liste_mots)
        
        max_length = PARAMS.MODEL.MAX_SENTENCE_LEN

        tous_les_ids = []
        tous_les_tags = []
        tous_les_masques = []
        for i, mot_entier in enumerate(liste_mots):
            mot_entier_encodé = self.tokenizer.encode_plus(
                mot_entier, padding='do_not_pad',
                max_length=None, truncation=False, add_special_tokens=False
            )
            
            tous_les_ids.extend(mot_entier_encodé['input_ids'])
            tous_les_tags.extend([tags[i]] * len(mot_entier_encodé['input_ids']))
            tous_les_masques.extend(mot_entier_encodé['attention_mask'])
        
        # Troncation (si nécessaire) :
        tous_les_ids = tous_les_ids[:max_length - 2]
        tous_les_tags = tous_les_tags[:max_length - 2]
        tous_les_masques = tous_les_masques[:max_length - 2]
        
        # Ajout des tokens de début et de fin :
        tous_les_ids = [self.special_token_start] + tous_les_ids + [self.special_token_end]
        tous_les_tags = [0] + tous_les_tags + [0]
        tous_les_masques = [1] + tous_les_masques + [1]
        
        # Ajout du padding :
        tous_les_ids = tous_les_ids + [self.PADDING_VALUE] * (max_length - len(tous_les_ids))
        tous_les_tags = tous_les_tags + [0] * (max_length - len(tous_les_tags))
        tous_les_masques = tous_les_masques + [0] * (max_length - len(tous_les_masques))

        return {
            "input_ids": torch.tensor(tous_les_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(tous_les_masques, dtype=torch.int64),
            "token_type_ids": torch.zeros(max_length, dtype=torch.int64),
            "labels": torch.tensor(tous_les_tags, dtype=torch.int64)
        }
