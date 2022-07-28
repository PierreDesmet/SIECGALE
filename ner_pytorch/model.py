import torch
import torch.nn as nn
from transformers import (BertForTokenClassification,
                          CamembertConfig, CamembertForTokenClassification)

from ner_pytorch.config.params import PARAMS


def loss_fn(y_pred, y_true, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = y_pred.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        y_true.view(-1),
        torch.tensor(lfn.ignore_index).type_as(y_true)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class EntityModel(nn.Module):
    def __init__(self, num_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        
        if PARAMS.LANGUAGE == 'en':
            self.bert = BertForTokenClassification.from_pretrained(
                PARAMS.PATHS_EN.MODEL,
                num_labels=num_tag,
                return_dict=False
            )
        elif PARAMS.LANGUAGE == 'fr':
            config = CamembertConfig.from_pretrained(
                PARAMS.PATHS_FR.MODEL,
                output_hidden_states=True
            )
            self.bert = CamembertForTokenClassification.from_pretrained(
                PARAMS.PATHS_FR.MODEL,
                num_labels=num_tag,
                local_files_only=True,
                return_dict=False
            )
        self.dropout = nn.Dropout(0.3)
        # il y aurait besoin de cette ligne si on utilisait le
        # que CamembertModel au lieu du CamembertForTokenClassification :
        # self.out_tag = nn.Linear(768, self.num_tag)
        
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 
            'labels': labels
        }
        loss, o1 = self.bert(**kwargs)
        bo_tag = self.dropout(o1)
        loss = loss_fn(bo_tag, labels, attention_mask, self.num_tag)
        return loss, bo_tag
