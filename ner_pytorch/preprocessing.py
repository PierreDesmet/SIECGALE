import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def process_data(df: pd.DataFrame):
    """
    Transforme le `df` en plusieurs numpy arrays :
    - les phrases
    - les NER
    - le LabelEncoder associé aux classes (Tag)
    """
    df = df.copy()
    # On force l'ordre :
    enc_tag = OrdinalEncoder(categories=[['O', 'B-org', 'I-org']])
    

    # Le '+1' sert à s'assurer que le modèle pourra distinguer le padding (classe 0)
    # de la première classe à prédire (classe 1). 
    # Cela samblait confirmé par Lukas Nielsen, quand il demande sur YT :
    # "Hi Abhishek, great tutorial as allways. One problem though; I was thinking that it might cause 
    # some issues that you extend the target_tag with 0's for the padding, since 0 is likely assigned 
    # as the encoded value for an actual tag. Am I mistaken in this?
    # Edit : en réalité, on s'en moque car peu importe les prédictions pour les tokens de padding, elles
    # ne contribuent pas à la loss (cf ner_pytorch.model.loss_fn).
    # df['Tag'] = enc_tag.fit_transform(df.Tag)  # + 1

    df['Tag'] = enc_tag.fit_transform(df[['Tag']])

    sentences = df.groupby('Sentence #')['Word'].apply(list).values
    tag = df.groupby("Sentence #")['Tag'].apply(list).values
    return sentences, tag, enc_tag