# Dropout Prediction with Conformal Prediction (Brazil & Portugal)

Ce dépôt contient des fonctions Python et des notebooks pour **prédire le décrochage** étudiant tout en **quantifiant l’incertitude** via la *conformal prediction* (classique et **Mondrian**), ainsi qu’une version séquentielle **SPCI** (Simple Prediction Conformal Intervals) hors‑ligne.

Deux jeux de données sont traités :
- **Brazil** : questionnaire (42 *Items*), méta‑données et cible binaire `dropout` (présent dans `comparative_data/data.xlsx`).  
- **Portugal** : notes par UE (colonnes finissant par `"(grade)"`) et variables statiques (p. ex. *Previous qualification*, *Curricular units 1st sem/2nd sem*), avec la cible binaire `dropout`.

> Les expériences principales sont reproduites par les notebooks : `pipeline_brazil.ipynb` et `pipeline_portugal.ipynb`.

---

## ⚙️ Installation

Python >= 3.12 recommandé. Créez un environnement virtuel puis installez les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install lightgbm mapie matplotlib numpy openpyxl pandas scikit-learn tqdm
```

> Si vous préférez un fichier `requirements.txt`, vous pouvez utiliser :  

> lightgbm
> mapie
> matplotlib
> numpy
> openpyxl
> pandas
> scikit-learn
> tqdm

---

## 📁 Structure du dépôt (extrait)

```
   ├─ brazil_functions.py
   ├─ pipeline_brazil.ipynb
   ├─ pipeline_portugal.ipynb
   ├─ portugal_functions.py
└─ comparative_data/
   ├─ data.xlsx
└─ __pycache__/
   ├─ brazil_functions.cpython-311.pyc
```

- `brazil_functions.py` : pipeline complet pour le jeu **Brazil** (clustering, calibration, prédiction conformale).  
- `portugal_functions.py` : pipeline pour **Portugal** (détection des colonnes `"(grade)"`, SPCI unilatéral et bilatéral).  
- `comparative_data/data.xlsx` : exemple de données *Brazil*.  
- `pipeline_brazil.ipynb`, `pipeline_portugal.ipynb` : scripts d’expérience.

---

## 🧠 Modèles et méthode

- **Classifieurs ponctuels** : `RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`.  
- **Régression/quantiles** : `RandomForestRegressor`, `GradientBoostingRegressor`, `LGBMRegressor`.  
- **Conformal prediction** : [`mapie.MapieClassifier`] avec **MondrianCP** (partition par clusters).  
- **SPCI (hors‑ligne)** : classes `OneSidedSPCI_LGBM_Offline` et `TwoSidedSPCI_RFQuant_Offline` pour estimer des **intervalles [L;U]** sur un flux (fenêtre de résidus de longueur *w*).  
- **Clustering** : `KMeans` après standardisation (`StandardScaler`).  
- **Enrichissement (minority oversampling)** : variantes *SMOTE‑like* par interpolation des plus proches voisins au sein de chaque cluster.

---

## 🗂️ Préparer vos données

### Brazil
- Colonnes attendues : `Instrument` (sera renommé en `email`), `dropout` (0/1), variables socio‑démographiques et **Items** `Item 1` … `Item 42`.  
- Exemple : charger `comparative_data/data.xlsx` puis :
  ```python
  import pandas as pd
  df = pd.read_excel("comparative_data/data.xlsx")
  ```

### Portugal
- Colonnes attendues : une colonne **identifiant** `email` (ou à fournir), la cible `dropout` (0/1), des colonnes de notes finissant par `"(grade)"`, et des variables statiques (ex. *Previous qualification*, *Curricular units 1st sem*, *Curricular units 2nd sem*).  
- Les fonctions détectent automatiquement `"(grade)"` et construisent les caractéristiques temporelles + statiques.

> Les valeurs manquantes sont imputées par `nan_fill` (défaut : Brazil 0, Portugal -1).

---

## 🚀 Utilisation rapide (API Python)

### Brazil
```python
from brazil_functions import run_analysis_brazil
import pandas as pd

df = pd.read_excel("comparative_data/data.xlsx")
df_detail, df_agg, y, clfs = run_analysis_brazil(
    df=df.drop(columns=["dropout"]),  # X
    y=df["dropout"],                  # y (0/1)
    alpha=0.05,                       # niveau pour la couverture (~95%)
    do_plot=False                     # True pour tracer quelques figures
)
print(df_agg.head())      # métriques agrégées
print(df_detail.head())   # résultats détaillés par méthode/modèle/cluster
```

### Portugal
```python
from portugal_functions import run_analysis_portugal
import pandas as pd

# df_portugal = pd.read_csv("vos_donnees_portugal.csv")
# y = df_portugal["dropout"]
# X = df_portugal.drop(columns=["dropout"])

df_detail, df_agg, y_all, clfs = run_analysis_portugal(
    df=X,
    y=y,
    alpha=0.05,
    nan_fill=-1.0,
    do_plot=False
)
```

**Sorties communes** :
- `df_detail` : résultats par *méthode* (`vanilla`, `Mondrian`, `SPCI`…), *modèle*, *cluster*, et *taille d’entraînement* (`n_projects`), avec **couverture** et **largeur** moyenne.  
- `df_agg` : agrégats globaux par méthode/modèle.  
- `y`/`y_all` : cible binaire alignée.  
- `clfs` : classifieurs calibrés par (méthode, cluster[, clé]) pour réutilisation.

---

## 🔧 Options importantes

- `alpha` : niveau d’erreur (p.ex. 0.05 ⇒ couverture visée ≈ 95 %).  
- `globe` : si `True`, agrège en plus des métriques « globales ».  
- `nan_fill` : constante d’imputation des NA.  
- Enrichissement : voir les helpers `oversample_minority_clusters` (Brazil) / `augment_minority_clusters` (Portugal).

---

## 📊 Reproduire les expériences

Ouvrez les notebooks :
- `pipeline_brazil.ipynb`
- `pipeline_portugal.ipynb`

Chaque notebook crée des variantes (*without cluster*, *without enrichment*, *with SMOTE*, *with GAN*†) et compare **couverture**/**largeur** des intervalles.  


---

## ✉️ Contact

Pour toute question : ouvrir une *issue* ou contacter les auteurs de ce dépôt.
