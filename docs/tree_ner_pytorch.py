"""
Simple tree-render from the command line.
Originally from: https://github.com/willmcgugan/rich/blob/master/examples/tree.py
"""

import os
import sys

from bs4 import BeautifulSoup
from pathlib import Path
from rich import print
from rich.console import Console
from rich.filesize import decimal
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree

WITH_CONSOLE_FORMAT = False
SRC = 'docs/tree_temp.md'
DEST = 'docs/Suivi/tree.md'

console = Console(record=True, width=100)

def walk_directory(directory: Path, tree: Tree) -> None:
    """Recursively build a Tree with directory contents."""
    # Sort dirs first then by filename
    paths = sorted(
        Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    for path in paths:
        # Remove hidden files
        if path.name.startswith("."):
            continue
        if path.name in ('__init__.py', 'tree_ner_pytorch.py'):
            continue
        if path.parts[-1] == "venv":
            continue
        if path.is_file():
            if path.parts[-1] == "reports":
                pass  # TODO !
        if path.is_dir():
            style = "dim" if path.name.startswith("__") else ""
            branch = tree.add(
                f"[bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}",
                style=style,
                guide_style=style,
            )
            walk_directory(path, branch)
        else:
            text_filename = Text(path.name, "green")
            text_filename.highlight_regex(r"\..*$", "bold red")
            text_filename.stylize(f"link file://{path}")
            file_size = path.stat().st_size
            text_filename.append(f" ({decimal(file_size)})", "blue")
            icon = "🐍 " if path.suffix == ".py" else "📄 "
            tree.add(Text(icon) + text_filename)


def replace_everything_between(file_path: str,
                               start_pat: str,
                               stop_pat: str,
                               new_content: str):
    """
    En attendant d'utiliser Jinja2, un petit utilitaire pour remplacer du
    fichier `file_path` tout le texte entre `start_pat` et `stop_pat` par `new_content`.
    """
    index = Path(file_path)
    content = index.read_text()
    start_pos = content.find(start_pat)
    stop_pos = content.find(stop_pat) + len(stop_pat)
    new_content = content[:start_pos] + new_content + content[stop_pos:]
    index.write_text(new_content)


# 1 - Création de l'arbre
try:
    directory = os.path.abspath(sys.argv[1])
    directory_name = Path(directory).name
except IndexError:
    print("[b]Usage:[/] python tree.py <DIRECTORY>")
else:
    tree = Tree(
        f":open_file_folder: [link file://{directory}]{directory_name}",
        guide_style="bold bright_blue",
    )
    walk_directory(Path(directory), tree)

console.print(tree)
console.print("")

# 2 - Enregistrement de l'arbre dans un markdown
if WITH_CONSOLE_FORMAT:
    CONSOLE_HTML_FORMAT = """\
    <pre style="font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">{code}</pre>
    """
    console.save_html(SRC, inline_styles=True, code_format=CONSOLE_HTML_FORMAT)
else:
    console.save_html(SRC, inline_styles=True)

# 3 - Extraction de l'arbre et envoi dans la documentation
all_text = Path(SRC).read_text()
text = BeautifulSoup(all_text, 'html.parser').find('pre').text
text = '<!-- <arborescence> -->' + text + '<!-- </arborescence> -->'
replace_everything_between(file_path=DEST,
                           start_pat='<!-- <arborescence> -->',
                           stop_pat='<!-- </arborescence> -->',
                           new_content=text)
os.remove(SRC)
