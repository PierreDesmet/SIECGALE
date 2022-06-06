import ast
import torch
from transformers import BertTokenizer

from ner_pytorch.config.params import PARAMS


class EntityDataset(torch.utils.data.Dataset):
    tokenizer = BertTokenizer.from_pretrained(PARAMS.PATHS.MODEL, 
                                              do_lower_case=PARAMS.MODEL.DO_LOWER_CASE)
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item: int):
        text = self.texts[item]
        tags = self.tags[item]

        ids = []
        target_tag =[]

        for i, s in enumerate(text):
            inputs = EntityDataset.tokenizer.encode(
                s,
                add_special_tokens=False
            )
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:PARAMS.MODEL.MAX_SENTENCE_LEN - 2]
        target_tag = target_tag[:PARAMS.MODEL.MAX_SENTENCE_LEN - 2]

        ids = [101] + ids + [102]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = PARAMS.MODEL.MAX_SENTENCE_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }
