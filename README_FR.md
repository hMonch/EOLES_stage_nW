# Eoles

Eoles est un modèle d'optimisation de l'investissement et de l'exploitation du système énergétique cherchant à minimiser le coût total tout en assurant une demande en énergie exogène (électricité, méthane, hydrogène). \
Voici une présentation d'une version antérieure du modèle : _http://www.centre-cired.fr/quel-mix-electrique-optimal-en-france-en-2050/ \
La plupart des versions du modèle, ainsi que des articles les utilisant, sont présentées dans https://www.centre-cired.fr/the-eoles-family-of-models/

Cette version (`modelEoles_multiN_v3_7.py`) est une réécriture **multi-nœuds** du modèle : elle optimise simultanément l'investissement et l'exploitation de la France et un ensemble de pays européens voisins (Espagne, Italie, Suisse, Allemagne, Belgique, Royaume-Uni, Pays-Bas, Irlande, Portugal), avec échanges d'électricité, de méthane et d'hydrogène entre pays. L'ancienne version mono-nœud (`modelEoles.py`, basée sur Pyomo) est conservée dans le dépôt à titre de référence historique.

---

### Installer le code et les dépendances

---

#### **Récupération du code :**




#### **Installation des dépendances**

Pour pouvoir lancer le modèle vous aurez besoin d'installer certaines dépendances dont ce programme à besoin pour fonctionner :

* **Python** :
Python est un langage de programmation interprété, utilisé avec [linopy](https://linopy.readthedocs.io/) (une surcouche de modélisation d'optimisation linéaire basée sur xarray) pour construire et résoudre Eoles. \
Vous pouvez télécharger la dernière version sur le site dédié : *https://www.python.org/downloads/* \
Ensuite il vous suffit de l'installer sur votre ordinateur. \
Si vous comptez installer Conda ou si vous avez installé Conda sur votre ordinateur, Python à des chances d'être déjà installé.
Le modèle nécessite python3. Nous recommandons python3.12 (également testé et validé avec python3.13 ; `environment.yml` cible 3.12).

* **Conda** ou **Pip** selon votre préférence :
Conda et Pip sont des gestionnaires de paquets pour Python. Conda est recommandé.
    * **Conda** \
    Vous pouvez retrouver toutes les informations nécéssaires pour télécharger et installer Conda ici: \
    _https://docs.conda.io/projects/conda/en/latest/user-guide/install/_ \
    __Attention à bien choisir la version de Conda en accord avec la version de Python !__ \
    Vous pouvez installer Miniconda qui est une version minimale de Conda,\
    cela vous permettra de ne pas installer tous les paquets compris dans Conda, \
    mais seulement ceux qui sont nécéssaires.
    * **Pip** \
    Vous pouvez retrouver toutes les informations nécéssaires pour télécharger et installer Pip  ici : \
    _https://pip.pypa.io/en/stable/installing/_ \
    Pip est également installé si vous avez installé Conda.

* Méthode d'installation Hadrien avec Conda :
Utilisez Anaconda Prompt pour cette tâche.
Déplacez vous jusqu'au dossier ou est situé l'environnement environment.yml
Créer l'environnement et installer les dépendances: ```conda env create -f environment.yml```
Activer l'environnement : ```conda activate env_EOLES_CIRED```


* Installer les dépendances avec **Conda**:
Déplacez-vous jusqu'au dossier de votre choix
Créer l'environnement et installer les dépendances: ```conda env create -f environment.yml```
Activer l'environnement : ```conda activate env_EOLES_CIRED```
Si vous souhaitez utiliser Jupyter Notebook :
Utiliser ```conda install -c anaconda ipykernel``` and ```python -m ipykernel install --user --name=env_EOLES_CIRED```
L'environnement sera alors disponible dans la liste des kernel.

* Installer les dépendances avec **Pip**:
Créer un environnement virtuel: ```python -m venv env_EOLES_CIRED```
Si vous tuilisez un autre nom pour l'environnement et avez prévu d'envoyer vos modifications vers le github, souvenez-vous d'exclure le dossier de l'environnement des commit.
Activer l'environnement :
Windows : ```env_EOLES_CIRED\Scripts\activate```
macOS/Linux: ```source env_EOLES_CIRED/bin/activate```
Installer les dépendances : ```pip install -r requirements.txt```

`requirements.txt` installe notamment `linopy`, `xarray`, `pandas`, `numpy`, `matplotlib`, `openpyxl` (lecture des fichiers Excel de scénario), ainsi que les deux solveurs ci-dessous.

* **Solveur** :
Le modèle est résolu avec le solveur **Gurobi** par défaut (`ModelEOLES.solve(solver_name="gurobi")`), bien plus rapide que les alternatives libres sur des problèmes de cette taille. \
Des licences gratuites sont mises à disposition pour les chercheurs et étudiants.
Pour utiliser Gurobi :
    * Se créer un compte et télécharger Gurobi Optimizer ici : _https://www.gurobi.com/downloads/_
    * Demander une licence académique gratuite : _https://www.gurobi.com/downloads/end-user-license-agreement-academic/_
    * Utiliser la commande ```grbgetkey``` pour importer sa licence, comme indiqué sur celle-ci. \
Pour utiliser Gurobi sur Inari : voir le README dédié.

Si Gurobi échoue à se lancer (licence absente, par exemple), `ModelEOLES.solve()` retente automatiquement avec **HiGHS** (`highspy`, solveur open-source installé par `requirements.txt`), ce qui permet de faire tourner le modèle sans licence, au prix d'un temps de résolution plus long sur les gros scénarios.

#### **Utilisation du modèle :**

Le modèle Eoles est écrit sous forme de classe `ModelEOLES` contenue dans `modelEoles_multiN_v3_7.py`. Les fonctions auxiliaires sont réparties par rôle dans quatre fichiers :
* `utils_io.py` — lecture de la configuration et des inputs (CSV) en `xr.DataArray`
* `utils_build.py` — fonctions utilisées pendant la construction du modèle (annuités)
* `utils_results.py` — extraction et mise en forme des résultats après résolution (coûts, bilans horaires, prix spot, prix de revient...)
* `utils_plots.py` — visualisation des résultats


**`example.py` est le point d'entrée à privilégier** pour un premier lancement, pour vérifier que tout est bien installé, ou pour lancer un scénario ponctuel : il construit, résout et exporte les résultats d'un cas minimal (FR + un pays voisin, une seule année climatique) en quelques minutes. C'est aussi le meilleur exemple pour écrire son propre script.

`run_batch.py` répond à un besoin différent : lancer le modèle sur un grand nombre d'années climatiques et de scénarios (typiquement sur un serveur de calcul, en tâche de fond). À réserver aux campagnes de calcul plus lourdes, une fois qu'on a vérifié via `example.py` que tout fonctionne.

##### Chaîne de génération des inputs

Toutes les données d'entrée partent d'un unique fichier Excel de scénario (`Scenario_data_EUR_plus.xlsx`) et de deux scripts :

1. **`reader.py`** lit les feuilles du fichier Excel (coûts, paramètres technologiques, capacités, potentiels de biogaz, budget carbone par pays...) et les écrit en CSV dans `inputs/` et `inputs/area_indexed/`.
2. **`create_demand/create_demand_complete.py`** (classe `DemandBuilder`) construit les courbes de charge horaires de demande électrique, méthane et hydrogène (France par région + pays européens) à partir des mêmes feuilles Excel, des chroniques de température et des profils journaliers types, et les écrit dans `inputs/time_varying_inputs/`.

Une fois ces deux scripts lancés, `config/config_multi_nodes.json` fait le lien entre le modèle et tous les fichiers CSV générés — il suffit ensuite de lancer `example.py` ou `run_batch.py` (ou d'écrire un script similaire) pour construire et résoudre le modèle.

---

### Données d'entrées

---

Les données d'entrée sont fournies dans les dossiers **inputs** (constantes, coûts, capacités par pays...) et **inputs/time_varying_inputs** (chroniques horaires de demande, de production renouvelable et hydraulique), générées comme décrit ci-dessus à partir de `Scenario_data_EUR_plus.xlsx`. \
Le chemin d'accès à chaque fichier de donnée peut être modifié dans `config/config_multi_nodes.json`. \

Le format attendu pour chaque type de donnée (constante ou profil, indexée par pays ou non) est clarifié par les fonctions de lecture associées dans `utils_io.py` (`read_constant_xr`, `read_profile_xr`, `read_links`).

---

#### **Exploitation des sorties :**

Deux notebooks avec des rôles différents sont à disposition afin de permettre l'analyse des résultats la plus simple et complète possible :

1. **`notebook_single_run_exploitation.ipynb`** : ce premier notebook permet de lancer des simulations individuelles du modèle EOLES sur une année climatique choisie. On peut notamment décider quels pays représenter, la présence de réserves ou l'année climatique.
Les cellules du notebook permettent une fois l'optimisation finie de visualiser les résultats, et notamment tout ce qui nécessite des données horaires (prix spot, dispatch sur une ou plusieurs semaine, demande résiduelle...). Il exploite des graphiques et méthodes de `utils_results.py` et `utils_plots.py`

2. **`notebook_batch_comparison`** : ce second notebook permet, à partir des
batch de résultats obtenus par lecture du fichier `run_batch.py`, de comparer les résultats du mix énergétique sur des années climatiques différentes.

La version originale de ce README a été écrite par Quentin Bustarret.\
Vous pourrez trouver les anciennes versions du modèle (code et articles pour lesquels elles ont été utilisées) sur cette page web : https://www.centre-cired.fr/the-eoles-family-of-models/
