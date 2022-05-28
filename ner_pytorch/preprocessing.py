import pandas as pd
from sklearn.preprocessing import LabelEncoder

def process_data(df: pd.DataFrame):
    """
    Transforme le `df` en plusieurs numpy arrays :
    - les phrases
    - les POS
    - les NER
    - les LabelEncoders associées aux classes (POS et TAG)
    """
    df = df.copy()
    enc_tag = LabelEncoder()

    # Le '+1' sert à s'assurer que le modèle pourra distinguer le padding (classe 0)
    # de la première classe à prédire (classe 1). Abhishek n'y avait pas pensé !
    df['Tag'] = enc_tag.fit_transform(df.Tag) + 1

    sentences = df.groupby('Sentence #')['Word'].apply(list).values
    tag = df.groupby("Sentence #")['Tag'].apply(list).values
    return sentences, tag, enc_tag