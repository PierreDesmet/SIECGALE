import codecs 
import json
from bs4 import BeautifulSoup
from pathlib import Path

from git import Repo


def git_repo_messages_to_json(output_name: str, repo_path='.'):
    """
    Extrait d'un repo un dictionnaire de la forme 
    {'11dc834': commit_msg, ...}
    """
    repo = Repo(repo_path)

    sha_to_msg = dict()
    for commit in repo.iter_commits():
        sha_to_msg[commit.hexsha] = commit.message

    json.dump(sha_to_msg, open(output_name, 'wt'))


def title_from_commit_msg(msg: str):
    """Le titre d'un commit à partir de son contenu"""
    return msg.split('<br>')[0]


def set_titre_du_commit_bold(msg):
    """Renvoie le commit msg mais avec le titre en gras"""
    titre_du_commit = title_from_commit_msg(msg)
    return msg.replace(titre_du_commit, f'<b>{titre_du_commit}</b>')


def sexify_git_history(git_history_path='git-history.md', 
                       template_url_gitlab='http://gitlab.groupe.generali.fr/gitlab/met-ta/datalab/parc/-/tree/{}',
                       output_path='resultat.html',
                       commit_messages_path='sha_to_msg.json'):
    """
    Code pour sexifier l'historique git :
    - les SHA des commits deviennent cliquables et renvoient vers le Gitlab,
    - les carrés vides deviennent l'icône de Gitlab,
    - un effet hoover permet d'afficher le détail du commit message au survol.
    """
    f_initial = codecs.open(git_history_path, 'r', 'utf-8').read()
    f_soup = BeautifulSoup(f_initial, 'html.parser')
    
    # Liens cliquables sur les SHA de commits :
    commit_lines = f_soup.find_all('td', "commit")
    for line in commit_lines:
        sha = line.contents[0]
        replacement = f'<a href="{template_url_gitlab}"' + ' target="blank">{}</a>'
        final_line = line.text.replace(sha, replacement.format(sha, sha))
        line.string = final_line


    # Suppression des carrés vides :
    icon_lines = f_soup.find_all('span', 'icon')
    replacement = ''
    for icon in icon_lines:
        icon.decompose()

    # Hover pour afficher le commit-message
    desc = json.load(open(commit_messages_path, 'rt'))

    new_dict = dict()
    for cle, valeur in desc.items():
        new_cle = cle[:7]
        new_dict[new_cle] = valeur.replace('\n', '<br>')
    
    for td in f_soup.find_all('td', 'd'):
        tr_block = td.find_parent()
        sha = tr_block['data-commitid']
        titre_du_commit = title_from_commit_msg(new_dict[sha])
        adding = f"""
          <span class="icon-box">
              <span class="icone icone__cow">
                <span class="icon">&nbsp;{titre_du_commit} ✔</span>
                  <div class="tooltipe tooltipe__cow">
                        {set_titre_du_commit_bold(new_dict[sha])}
                  </div>
              </span>
          </span>"""
        td.string = adding

    style = """
    <head>
        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
        <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.7.0/css/all.css"
            integrity="sha384-lZN37f5QGtY3VHgisS14W3ExzMWZxybE1SJSEsQp9S+oqd12jhcu+A56Ebc1zFSJ" crossorigin="anonymous">

    """

    
    # Export : 
    f_soup.string = str(f_soup.prettify(formatter=None)).replace('<head>', style)
    html = f_soup.prettify('utf-8', formatter=None)
    Path(output_path).write_text(f_soup.text)
    

git_history_path = 'site/Suivi/git-history.html'
template_url_gitlab = 'http://gitlab.groupe.generali.fr/gitlab/met-ta/datalab/parc/-/tree/{}'
output_path = 'site/Suivi/git-history.html'
commit_messages_path = 'docs/Suivi/sha_to_msg.json'
repo_path = '.'

git_repo_messages_to_json(output_name=commit_messages_path, repo_path=repo_path)
f_soup = sexify_git_history(git_history_path=git_history_path, 
                   template_url_gitlab=template_url_gitlab,
                   output_path=output_path,
                   commit_messages_path=commit_messages_path)
