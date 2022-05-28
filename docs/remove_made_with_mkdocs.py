from bs4 import BeautifulSoup, Tag
from pathlib import Path


def made_with_mkdocs(tag: Tag):
    return tag and tag.text == 'Documentation générée avec MkDocs.'

def soupify_and_remove_made_with_mkdocs(html_content: str):
    bs = BeautifulSoup(html_content, 'html.parser')
    replace = BeautifulSoup('<p>Documentation générée par Pierre DESMET<p>', 'html.parser')
    if bs.find(made_with_mkdocs):
        bs.find(made_with_mkdocs).replace_with(replace)
    return bs

def remove_made_with_mkdocs_from_documentation(site_path: str):
    for html_file in Path().rglob('*.html'):
        modified_bs = soupify_and_remove_made_with_mkdocs(html_content=html_file.read_text())
        _ = html_file.write_text(str(modified_bs))
        
remove_made_with_mkdocs_from_documentation('../site/')