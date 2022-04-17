import datetime

import faker
import numpy as np
import pendulum
import spacy
from string import Template


nlp = spacy.load("fr_core_news_sm")
fake = faker.Faker('fr_FR')


def french_date(date=None):
    """
    Par défaut la date du jour.
    >>> french_today()
    'Dimanche 04 avril 2021 à 09h48'
    """
    if date is None:
        today = datetime.datetime.today()
        today = pendulum.datetime(today.year, today.month,
                                  today.day, today.hour, today.minute)
        today = today.in_tz(tz='Europe/Paris')
        today = today.format('dddd DD MMMM YYYY [à] HH[h]mm', locale='fr')
        return today.capitalize()
    date = pendulum.datetime(date.year, date.month, date.day)
    return date.in_tz(tz='Europe/Paris').format('dddd DD MMMM YYYY', locale='fr')


def génère_faux_articles(n: int):
    """
    Génère `n` faux articles dans une liste. Chaque article est un dictionnaire avec :
    - une date
    - un contenu
    - l'entreprise trouvée
    - si l'entreprise est un client | ancien client | prospect refusé | autre
    """
    
    article_template = """
    Le feu s'est déclaré dans la nuit du $date, dans le quartier de la Moutonne.
    Une enquête est ouverte pour déterminer les causes de l'incendie.
    Un bâtiment de 650m2 de l'entreprise $entreprise a été ravagé par les flammes à La Crau, rapporte La Provence.
    """.strip()

    articles = []
    for i in range(n):
        date = fake.date_this_year()
        entreprise = fake.company()
        siren = fake.siren()
        choix_statut_entreprise = ['Client', 'Ancien client', 'Prospect',
                                    'Entreprise sans interaction avec Generali']
        statut = np.random.choice(choix_statut_entreprise)
        article = {
            'date_article': date,
            'entreprise': entreprise,
            'siren': siren,
            'statut_entreprise': statut,
            'contenu': Template(article_template).substitute(entreprise=entreprise, date=french_date(date)),
        }
        articles.append(article)
    
    return articles


def stylise_contenu(text: str):
    """Met en surbrillance les entités nommées"""
    doc = nlp(text)
    html = spacy.displacy.render([doc], style="ent", jupyter=False, minify=True)
    return html.replace('\n', '')


def stylise_siren(row: str):
    """Permet d'avoir les SIREN cliquables vers société.com"""
    siren, entreprise = row['siren'], row['entreprise']
    if isinstance(siren, str):
        siren = siren.replace(' ', '')
    lien = f'https://www.societe.com/cgi-bin/search?champs={siren}'
    return f"<a href='{lien}'>{entreprise}</a>"
