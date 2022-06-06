import torch
import torch.nn as nn
#from transformers import BertModel
from pytorch_pretrained_bert import BertForTokenClassification

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
        # self.bert = BertModel.from_pretrained(PARAMS.PATHS.MODEL, return_dict=False) 'bert-based-uncased'
        self.bert = BertForTokenClassification.from_pretrained(PARAMS.PATHS.MODEL, num_labels=num_tag)
        self.dropout = nn.Dropout(0.3)
        # self.out_tag = nn.Linear(768, self.num_tag)
    
    def forward(self, ids, mask, token_type_ids, target_tag):        
        o1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo_tag = self.dropout(o1)
        # tag = self.out_tag(bo_tag)
        # loss = loss_fn(tag, target_tag, mask, self.num_tag)
        loss = loss_fn(bo_tag, target_tag, mask, self.num_tag)
        return bo_tag, loss
