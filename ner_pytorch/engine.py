import torch
from tqdm import tqdm

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
    model.eval()
    final_loss = 0
    for num_batch, batch in enumerate(data_loader):
        
        if device != 'cpu':
            for k, v in batch.items():
                batch[k] = v.to(device)
        
        _, loss = model(**batch)
        final_loss += loss.item()
    
    return final_loss / len(data_loader)
