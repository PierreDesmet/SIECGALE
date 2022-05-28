#########################################################
# Source : https://calmcode.io/makefiles/projects.html  #
# Les Makefiles conviennent sous UNIX mais pas Windows. #
# Pour les exécuter : make + nom-de-la-commande.        #
#########################################################

# PHONY: ma_variable --> même s'il existe déjà un dossier ma_variable, la
# commande ma_variable sera exécutée.
.PHONY: docs

help:
	@echo "available methods:"
	@echo " - install  : install the requirements.txt and pre-commit hooks"
	@echo " - config   : make git messages colorful"
	@echo " - tree     : create the tree of the project and append it to README.md"
	@echo " - clean    : remove unnecessary folders from working space"

build: test tree
	rm -rf site
	mkdocs build --no-directory-urls
	python docs/remove_made_with_mkdocs.py
	python docs/sexify_git_history.py
	# cp -R site/ un_point_de_montage

clean:
	rm -rf **/.ipynb_checkpoints
	rm -rf **/**/.ipynb_checkpoints
	rm -rf **/__pycache__
	rm -rf **/**/__pycache__
	rm -rf **/.pytest_cache
	rm -rf **/**/.pytest_cache	
	rm -rf **/.DS_Store
	rm -rf **/**/.DS_Store

config:
	git config --global color.diff auto
	git config --global color.status auto
	git config --global color.branch auto

docs:
	echo 'TODO !'

git-history:
	git log --all --date-order --pretty="%h|%p|%s"
	echo "\nhttps://bit-booster.com/graph.html"

install: config
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	pre-commit install
	pre-commit autoupdate

test:
	python -m pytest --version
	python -m pytest --pdb --disable-warnings --cov=ner_pytorch --cov-report="html:docs/Suivi/reports"
	# python -m pytest --doctest-modules --doctest-continue-on-failure

tree: clean
	python docs/tree_ner_pytorch.py .
