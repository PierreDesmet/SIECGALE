import pandas as pd
import streamlit as st
from PIL import Image
from sqlite_utils import Database

from script import génère_faux_articles, stylise_contenu, stylise_siren


# Configuration
logo = Image.open("logo.png")
st.set_page_config(layout="wide", 
                   page_title='SIECGALE',
                   page_icon=logo)


col1, col2 = st.columns([1, 20])
with col1:
    st.write('')
    st.write('')
    st.image('logo.png', width=60)
with col2:
    st.markdown(f'<h1><font color="#169147">SIECGALE</font></h1>',
                unsafe_allow_html=True)


# Inputs utilisateur
début = st.date_input('Du :')
fin = st.date_input('Au :')


# Accès à la BDD
db = Database("base_sinistre.db")
requête = f"""
SELECT *
FROM sinistres
WHERE date_article > '{début}' AND date_article < '{fin}'
ORDER BY date_article DESC
"""
articles_df = pd.DataFrame(db.query(requête))

nb_articles = articles_df.shape[0]
s = 's' * (nb_articles > 1)

texte = f'{nb_articles} résultat{s} trouvé{s}. '
texte += 'Les 5 premiers :' * (nb_articles > 5)
st.write(texte)


if not articles_df.empty:
    articles_df_fmt = articles_df.copy()
    articles_df_fmt['contenu'] = articles_df_fmt.contenu.apply(stylise_contenu)
    articles_df_fmt['entreprise'] = articles_df_fmt.apply(stylise_siren, axis=1)
    
    st.markdown(
        articles_df_fmt.drop('siren', axis=1)\
            .head()\
            .to_html(escape=False, render_links=True),
        unsafe_allow_html=True
    )
    

# Téléchargement du csv :
@st.cache
def convert_df(df, limit: int = 100_000):
    return df.head(limit).to_csv().encode('utf-8')

st.download_button(
   "Exporter en csv",
   convert_df(articles_df),
   f"Articles entre le {début} et le {fin}.csv",
   "text/csv",
   key='download-csv'
)
