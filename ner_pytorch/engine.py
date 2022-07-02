import pandas as pd
import torch
from tqdm import tqdm
from torch import nn

from ner_pytorch.model import loss_fn
from ner_pytorch.config.params import PARAMS


def train_fn(data_loader, model, optimizer, device, scheduler, 
             pbar=None, num_epoch=None):
    """Entraîne pendant UNE epoch"""
    model.train()
    final_loss = 0
    for num_batch, batch in enumerate(data_loader):
        if device != 'cpu':
            for k, v in batch.items():
                batch[k] = v.to(device)
        
        optimizer.zero_grad()
        batch_pred, loss = model(**batch)
        
        if num_batch % 10 == 0:
            with torch.no_grad():
                batch_loss = loss_fn(batch_pred, batch['target_tag'],
                                     batch['mask'], num_labels=PARAMS.NUM_LABELS)
                print(f'Batch #{num_batch} : loss = {batch_loss:.6f}')
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                       max_norm=PARAMS.MODEL.GRAD_MAX_NORM)
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
        if pbar is not None:
            pbar.set_description(f"Epoch {num_epoch}, batch {num_batch}")
    
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    """Evalue le modèle en cours d'entraînement sur une epoch"""
    with torch.no_grad():
        model.eval()
        final_loss = 0
        for num_batch, batch in enumerate(data_loader):

            if device != 'cpu':
                for k, v in batch.items():
                    batch[k] = v.to(device)

            _, loss = model(**batch)
            final_loss += loss.item()

    return final_loss / len(data_loader)


def prédit_single_doc(doc: str, model, tokenizer, max_length: int, label_enc):
    """
    Code pour prédire un unique document.
    
    Usage :
    >>> doc = "Le feu s'est déclaré en toute fin d'après-midi au sein de l'entreprise Génération piscine, spécialisée dans la fabrication de coques  polyester de piscines"
    >>> df = prédit_single_doc(doc, tokenizer=EntityDataset.tokenizer, 
                               max_length=PARAMS.MODEL.MAX_SENTENCE_LEN, 
                               label_enc=ordinal_enc_NER)
    """
    tokens_encodés = tokenizer.encode_plus(
        doc, padding='max_length', max_length=max_length
    )

    single_example = {
        'ids': torch.tensor(tokens_encodés['input_ids'], dtype=torch.int64).unsqueeze(0),
        'mask': torch.tensor(tokens_encodés['attention_mask'], dtype=torch.int64).unsqueeze(0),
        'token_type_ids': torch.zeros(max_length, dtype=torch.int64).unsqueeze(0),
        'target_tag': torch.zeros(max_length, dtype=torch.int64).unsqueeze(0)
    }
    with torch.no_grad():
        _ = model.eval()
        output, loss = model(**single_example)

    # Retrait des tokens qui ne contribuent pas à la loss (CLS, SEP, PAD) :
    output = output.squeeze()[single_example['mask'][0] == 1]

    predictions_probas = nn.functional.softmax(output, dim=1).detach().squeeze()
    predictions_probas, predictions_classes = torch.max(predictions_probas, dim=1)

    nb_tokens = single_example['mask'].sum().item()
    ids = single_example['ids'].squeeze()[:nb_tokens]

    résultat = pd.DataFrame({
        'token_text': [tokenizer.decode([token]) for token in ids],
        'token_id': ids,
        'pred_code': predictions_classes,
        'pred_label': label_enc.inverse_transform(predictions_classes.reshape(-1, 1)).squeeze(),
        'y_true': label_enc.inverse_transform(single_example['target_tag'].squeeze()[:nb_tokens].reshape(-1, 1)).squeeze(),
        'class_proba': predictions_probas
    })
    return résultat.set_index('token_text')
