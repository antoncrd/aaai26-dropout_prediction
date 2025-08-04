from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, Union, List
from sklearn.base import clone
from mapie.mondrian import MondrianCP
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from mapie.classification import MapieClassifier
from mapie.metrics import (
    classification_coverage_score,
    classification_mean_width_score
)
from sklearn.ensemble import RandomForestRegressor
from collections import deque
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans

def assign_clusters_with_min_size(
    df: pd.DataFrame,
    n_clusters: int = 20,
    min_cluster_size: int = 150,
    drop_cols: list = ['email', 'dropout', 'source']
) -> pd.DataFrame:
    """
    Effectue un clustering KMeans avec normalisation et réaffecte les points des petits clusters
    vers le cluster valide le plus proche.

    Args:
        df (pd.DataFrame): Données d'entrée avec observations en lignes.
        n_clusters (int): Nombre initial de clusters pour KMeans.
        min_cluster_size (int): Taille minimale acceptable pour un cluster.
        drop_cols (list): Colonnes à exclure avant normalisation.

    Returns:
        pd.DataFrame: Copie de df avec une colonne 'cluster' assignée.
    """
    # 1. Prétraitement
    X = df.drop(columns=drop_cols).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Clustering initial
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    # 3. Taille des clusters
    sizes = pd.Series(labels).value_counts().sort_index()
    small_clusters = sizes[sizes < min_cluster_size].index.tolist()

    # 4. Réaffectation des points de petits clusters
    if small_clusters:
        print(f"Clusters trop petits à réaffecter : {small_clusters}")
        for sc in small_clusters:
            idxs = np.where(labels == sc)[0]
            for i in idxs:
                dists = np.linalg.norm(X_scaled[i] - centroids, axis=1)
                dists[sc] = np.inf
                for s in small_clusters:
                    dists[s] = np.inf
                labels[i] = int(np.argmin(dists))

        sizes = pd.Series(labels).value_counts().sort_index()
        print("Nouvelles tailles de clusters :\n", sizes)

    # 5. Ajout des labels au dataframe d'origine
    df_out = df.copy()
    df_out['cluster'] = labels
    return df_out

def augment_minority_clusters(
    df: pd.DataFrame,
    grade_suffix: str = '(grade)',
    numeric_extras: List[str] = ['Unemployment rate', 'Inflation rate', 'GDP'],
    n_by_cluster_min: int = 100,
    k_neighbors: int = 20,
    min_quantile: float = 0.25,
    drop_cols: List[str] = ['email', 'source']
) -> pd.DataFrame:
    """
    Augmente les clusters minoritaires d'un DataFrame par interpolation de voisins proches,
    avec génération contrôlée des variables numériques et catégorielles.

    Args:
        df (pd.DataFrame): Données d'origine, avec une colonne 'cluster'.
        grade_suffix (str): Suffixe caractérisant les colonnes de type "note".
        numeric_extras (list): Colonnes numériques supplémentaires à inclure.
        min_quantile (float): Seuil de quantile pour définir les clusters minoritaires.
        n_new_per (int): Nombre de points synthétiques par observation existante.
        max_neighbors (int): Nombre max de voisins à utiliser dans NearestNeighbors.
        drop_cols (list): Colonnes à exclure de l'espace des features.

    Returns:
        pd.DataFrame: Données concaténées (réelles + synthétiques).
    """
    # --- 0) Préparation ---
    X = df.drop(columns=drop_cols).fillna(0).reset_index(drop=True).copy()
    clusters = X['cluster'].values
    feature_cols = X.columns.drop('cluster')

    # Séparation numériques / catégorielles
    num_cols = [c for c in feature_cols if c.endswith(grade_suffix)] + [c for c in numeric_extras if c in feature_cols]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    cluster_col='cluster'
    # --- 1) Probabilités conditionnelles des colonnes binaires/catégorielles par cluster ---
    cluster_cat_probs = {}
    for cl, grp in X.groupby(cluster_col):
        cluster_cat_probs[cl] = {
            c: grp[c].value_counts(normalize=True).to_dict()
            for c in cat_cols
        }

    # 2) Détermination des clusters minoritaires
    counts = pd.Series(clusters).value_counts()
    threshold = counts.quantile(min_quantile)
    minor_cs = counts[counts < threshold].index.tolist()

    # 3) Génération des échantillons synthétiques
    rows_aug = []
    num_idx = [feature_cols.get_loc(c) for c in num_cols]

    for cl in minor_cs:
        idx = np.where(clusters == cl)[0]
        Xk = X.loc[idx, feature_cols].values
        n_new_per = n_by_cluster_min // len(idx)

        # Nombre de voisins effectifs (hors soi-même)
        k_eff = max(1, min(k_neighbors, len(idx) - 1))
        n_nbrs = min(k_eff + 1, len(Xk) - 1)

        # Duplication si trop peu de points
        if len(Xk) < 2:
            for i in idx:
                for _ in range(n_new_per):
                    rows_aug.append(X.loc[i, feature_cols].to_dict())
            continue

        nbrs = NearestNeighbors(n_neighbors=n_nbrs, metric='euclidean')
        nbrs.fit(Xk[:, num_idx])
        neigh_idxs = nbrs.kneighbors(return_distance=False)

        for i, xi in enumerate(Xk):
            for _ in range(n_new_per):
                nbr_list = [j for j in neigh_idxs[i] if j != i]
                j = np.random.choice(nbr_list)
                xj = Xk[j]

                lam = np.random.rand()
                num_new = xi[num_idx] + lam * (xj[num_idx] - xi[num_idx])

                cat_new = {
                    c: np.random.choice(
                        list(cluster_cat_probs[cl][c].keys()),
                        p=list(cluster_cat_probs[cl][c].values())
                    )
                    for c in cat_cols
                }

                new_row = {c: num_new[k] for k, c in enumerate(num_cols)}
                new_row.update(cat_new)
                new_row[cluster_col] = cl
                rows_aug.append(new_row)

    # 4) Assemblage du DataFrame final
    df_synth = pd.DataFrame(rows_aug)
    df_synth['source'] = 'synth'
    df_synth['email'] = np.nan 

    df_orig = df.copy()

    df_final = pd.concat([df_orig, df_synth], ignore_index=True)
    return df_final

def run_analysis_portugal(
    csv_file: Path | None = None,
    df: pd.DataFrame | None = None,
    y: Union[pd.Series, np.ndarray] | None = None,
    *,
    alpha: float = 0.05,
    n_rendus: int = 3,
    quantile_cut: float = 0.15,
    nan_fill: float = -1.0,
    do_plot: bool = False,
    globe: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[tuple[str,int], MapieClassifier]]:
    # ── 0) Chargement de df0 ─────────────────────────────────────
    if df is None:
        if csv_file is None:
            raise ValueError("Either df or csv_file must be provided.")
        df0 = pd.read_csv(csv_file)
    else:
        df0 = df.copy()

    if "email" not in df0.columns:
        raise ValueError("Must contain an 'email' column.")

    # on renomme pour compatibilité interne
    df0 = df0.fillna(nan_fill).rename(columns={"email": "student_id"})

    # ── 1) Construction de y_all ─────────────────────────────────
    if y is None:
        # même code que précédemment pour construire y_all
        mark_cols = [c for c in df0.columns if c.endswith("mark")][::-1]

        def last_marks(row):
            vals, cols = [], []
            for c in mark_cols:
                v = row[c]
                if v >= 0:
                    vals.append(v)
                    cols.append(c)
                    if len(vals) == n_rendus:
                        break
            return pd.Series({"cols": cols, "vals": vals})

        tmp = df0.apply(last_marks, axis=1)
        df0[["last_cols", "last_vals"]] = tmp

        means = {c: df0.loc[df0[c] >= 0, c].mean() for c in mark_cols}

        def norm_avg(vals, cols):
            if not vals:
                return np.nan
            return np.mean([v / means[c] for v, c in zip(vals, cols)])

        df0["norm_last_n"] = df0.apply(
            lambda r: norm_avg(r["last_vals"], r["last_cols"]), axis=1
        )

        thresh = df0["norm_last_n"].quantile(quantile_cut)
        y_all = (df0["norm_last_n"] < thresh).astype(int)
        print(
            f"Quantile {quantile_cut:.2f} = {thresh:.3f} → {y_all.mean() * 100:.1f}% positives"
        )
    else:
        # si y fourni, on s'assure que c'est une Series alignée
        y_all = pd.Series(y).reset_index(drop=True).astype(int)

    # array pour stratify et autres usages numpy
    y_all_arr = y_all.values

    # ── 2) Préparation des features & clusters ────────────────────
    has_cluster = "cluster" in df0.columns
    clusters_all = df0.get("cluster", pd.Series(0, index=df0.index)).astype(int).values

    # curri_cols = [c for c in df0.columns if c.endswith("(grade)")]
    curri_cols = ['Previous qualification', "Curricular units 1st sem", "Curricular units 2nd sem"]
    prefixes = curri_cols.copy()
    dyn_cols = [
        col for col in df0.columns
        if any(col.startswith(pref) for pref in prefixes)
        ]
    static_cols = [
    c for c in df0.columns
    if c not in dyn_cols + ["student_id", "email", "dropout", "source", "cluster"]
    ]

    # 3) Adapter build_X pour ne prendre QUE ces items
    def build_X(df_sub: pd.DataFrame, n: int) -> np.ndarray:
        # on garde student_id + les n premiers items
        dyn_cols_ = [
        col for col in df0.columns
        if any(col.startswith(pref) for pref in prefixes[:n])
        ]
        keep = ["student_id"] + dyn_cols_ + static_cols
        return df_sub[keep].set_index("student_id").values
    # ── 3) Boucle conformal Mondrian ─────────────────────────────
    MODELS = {
        "RF": RandomForestClassifier(
            n_estimators=1000,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "LR": LogisticRegression(
            max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "GB": GradientBoostingClassifier(random_state=42),
    }

    records: list[dict] = []
    trained_clfs: Dict[tuple[str, int], MapieClassifier] = {}

    for name, base_clf in MODELS.items():
        for n in tqdm(range(1, len(prefixes) + 1), desc=name):
            clf = clone(base_clf)
            idx_all = df0.index.to_numpy()

            # 3.a) split (train+cal) / test
            idx_tmp, idx_test, y_tmp, y_test, cl_tmp, cl_test = train_test_split(
                idx_all,
                y_all_arr,
                clusters_all,
                test_size=0.20,
                stratify=y_all_arr,
                random_state=42,
            )

            # 3.b) split train / cal
            idx_tr, idx_cal, y_tr, y_cal, cl_tr, cl_cal = train_test_split(
                idx_tmp,
                y_tmp,
                cl_tmp,
                test_size=0.20 / 0.80,
                stratify=y_tmp,
                random_state=42,
            )

            # 3.c) construction des X
            X_tr = build_X(df0.loc[idx_tr], n)
            X_cal = build_X(df0.loc[idx_cal], n)
            X_test = build_X(df0.loc[idx_test], n)

            # 3.d) entraînement + calibration
            clf.fit(X_tr, y_tr)
            calib = CalibratedClassifierCV(clf, cv="prefit", method="sigmoid")
            calib.fit(X_cal, y_cal)

            mapie_global = MapieClassifier(estimator=calib, method="lac", cv="prefit").fit(X_cal, y_cal)
            base_mapie = MapieClassifier(estimator=calib, method="lac", cv="prefit")

            yp_glob, pset_glob = mapie_global.predict(X_test, alpha=alpha)
            pset_glob = pset_glob[:,:,0]
            # 3.e) préparation Mondrian sur clusters valides
            ser = pd.Series(y_cal, index=pd.Index(cl_cal, name="cluster"))
            valid_clusters = (
                ser.groupby("cluster")
                .nunique()
                .loc[lambda s: s >= 2]
                .index.values
            )
            mask_cal_valid = np.isin(cl_cal, valid_clusters)

            mond_mapie = MondrianCP(
                mapie_estimator=base_mapie
            )
            mond_mapie.fit(
                X_cal[mask_cal_valid],
                y_cal[mask_cal_valid],
                partition=cl_cal[mask_cal_valid],
            )

            # 3.f) prédiction conformal
            mask_valid = np.isin(cl_test, valid_clusters)
            print(valid_clusters)
            mask_invalid = ~mask_valid
            y_pred = np.empty_like(y_test)
            pset = np.empty((len(y_test), 2), dtype=bool)

            if mask_valid.any():
                try:
                    yp_v, yps_v = mond_mapie.predict(
                        X_test[mask_valid], alpha=alpha, partition=cl_test[mask_valid]
                    )
                except ValueError:
                    yp_v, yps_v = mapie_global.predict(
                        X_test[mask_valid], alpha=alpha
                    )
                y_pred[mask_valid] = yp_v
                pset[mask_valid] = yps_v[:, :, 0]

            if mask_invalid.any():
                if globe:
                    yp_g, yps_g = mapie_global.predict(
                        X_test[mask_invalid], alpha=alpha
                    )
                    y_pred[mask_invalid] = yp_g
                    pset[mask_invalid] = yps_g[:, :, 0]
                else:
                    for k in np.unique(cl_test[mask_invalid]):
                        mask_k = mask_invalid & (cl_test == k)
                        y_k = np.unique(y_cal[cl_cal == k])[0]
                        y_pred[mask_k] = y_k
                        pset_bool_k = np.zeros((mask_k.sum(), pset.shape[1]), dtype=bool)
                        pset_bool_k[:, int(y_k)] = True
                        pset[mask_k] = pset_bool_k

            trained_clfs[(name, n)] = mond_mapie

            # 3.g) enregistrement des scores
            mask_real = df0.loc[idx_test, "source"] == "real"
            cov_all = classification_coverage_score(y_test[mask_real], pset[mask_real])
            width_all = classification_mean_width_score(pset[mask_real])
            records.append(
                {   "method": "mondrian",
                    "model": name,
                    "n_projects": n,
                    "cluster": -1,
                    "coverage": cov_all,
                    "width": width_all,
                }
            )
            cov_glob = classification_coverage_score(y_test[mask_real], pset_glob[mask_real])
            wid_glob = classification_mean_width_score(pset_glob[mask_real])
            records.append(
                {   "method": "vanilla",
                    "model": name,
                    "n_projects": n,
                    "cluster": -1,
                    "coverage": cov_glob,
                    "width": wid_glob,
                }
            )

    # ── 4) Agrégation des résultats ────────────────────────────────
    df_detail = pd.DataFrame.from_records(records)
    df_agg_cluster = (
        df_detail.groupby(["model", "cluster"])
        .agg(mean_coverage=("coverage", "mean"), mean_width=("width", "mean"))
        .reset_index()
    )
    df_agg_global = (
        df_detail.groupby("model")
        .agg(mean_coverage=("coverage", "mean"), mean_width=("width", "mean"))
        .reset_index()
        .assign(cluster="ALL")
    )
    df_agg = pd.concat([df_agg_cluster, df_agg_global], ignore_index=True)

    # ── 5) Plots (inchangés) ───────────────────────────────────────
    if do_plot:
        df_detail_1 = df_detail.fillna(-1)
        clusters = df_detail_1["cluster"].unique()
        metric_map = [("coverage", "Couverture"), ("width", "Width")]
        for metric, label in metric_map:
            for c in clusters:
                if c == -1:
                    mask = df_detail_1["cluster"].isna() | (df_detail_1["cluster"] == -1)
                    title_cl = "global-split"
                else:
                    mask = df_detail_1["cluster"] == c
                    title_cl = f"cluster {c}"
                if not mask.any():
                    continue
                plt.figure()
                for model, grp in df_detail_1[mask].groupby("model"):
                    grp_sorted = grp.sort_values("n_projects")
                    plt.plot(
                        grp_sorted["n_projects"],
                        grp_sorted[metric],
                        label=model,
                        marker="o",
                        linewidth=2,
                    )
                plt.xlabel("Nombre de projets")
                plt.ylabel(label)
                plt.title(f"{label} (α={alpha}) – {title_cl}")
                plt.grid(True)
                plt.legend()
                plt.xticks(sorted(df_detail["n_projects"].unique()))
                plt.tight_layout()
                plt.show()

    return df_detail, df_agg, y_all, trained_clfs

class TwoSidedSPCI_RFQuant_Offline:
    """
    Intervalle bilatéral SPCI (équations 10‑11) estimé hors‑ligne :

        [  f̂(Xₜ) + Q̂_{b,t}(β̂) ,  f̂(Xₜ) + Q̂_{b,t}(1‑α+β̂)  ]
        β̂ = argmin_{β∈[0,α]} ( Q̂_{b,t}(1‑α+β) − Q̂_{b,t}(β) )

    Les quantiles Q̂ sont prédits par une forêt de régression quantile
    entraînée sur les w derniers résidus.

    Hyp. : scikit‑learn ≥ 1.0 (mais même sans l’API quantile native,
           on calcule les quantiles en agrégeant les prédictions
           individuelles des arbres comme le fait scikit‑garden).
    """

    def __init__(self,
                 alpha=0.10,
                 w=400,
                 n_estimators=200,
                 n_estimators_q=300,
                 max_depth=-1,
                 random_state=0,
                 n_grid=101):
        self.alpha      = float(alpha)
        self.w          = int(w)
        self.n_grid     = int(n_grid)

        # --- 1) modèle ponctuel f̂
        self.base_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None if max_depth == -1 else max_depth,
            random_state=random_state,
        )

        # --- 2) modèle quantile sur les résidus
        self.rf_quant = RandomForestRegressor(
            n_estimators=n_estimators_q,
            max_depth=None if max_depth == -1 else max_depth,
            random_state=random_state + 1,
        )

        self.res_buf  = deque()   # buffer circulaire des résidus
        self.is_ready = False

    # ------------------------------------------------------------------ #
    #  Apprentissage                                                     #
    # ------------------------------------------------------------------ #
    def fit(self, X_train, y_train):
        """Entraîne le modèle ponctuel puis la forêt quantile."""
        # 1) f̂
        self.base_rf.fit(X_train, y_train)
        residuals = y_train - self.base_rf.predict(X_train)
        self.res_buf.extend(residuals)

        if len(self.res_buf) < self.w + 5:
            raise ValueError(f"Besoin d’au moins w+5={self.w+5} résidus ; "
                             f"buffer={len(self.res_buf)}")

        # 2) fenêtre glissante des résidus (caractéristiques) ────────
        R  = np.asarray(self.res_buf, dtype=float)
        Y  = R[self.w:]                                       # cible = r_t'
        Xq = np.array([R[i-self.w:i] for i in range(self.w, len(R))])

        # 3) forêt de régression quantile
        self.rf_quant.fit(Xq, Y)

        self.is_ready = True
        return self

    # ------------------------------------------------------------------ #
    #  Méthodes utilitaires                                              #
    # ------------------------------------------------------------------ #
    def _window_residuals(self):
        """Renvoie les w derniers résidus (paddés à 0 si nécessaire)."""
        buf = list(self.res_buf)
        if len(buf) < self.w:
            buf = [0.0] * (self.w - len(buf)) + buf
        return np.asarray(buf[-self.w:], dtype=float).reshape(1, -1)

    @staticmethod
    def _rf_quantile(tree_preds, q):
        """Quantile empirique des prédictions des arbres (0 ≤ q ≤ 1)."""
        return float(np.quantile(tree_preds, q, method="linear"))

    # ------------------------------------------------------------------ #
    #  Prédiction d’intervalle                                           #
    # ------------------------------------------------------------------ #
    def predict_interval(self, x_t):
        """
        Renvoie (Lₜ, Uₜ) selon SPCI, avec β̂ choisi pour largeur minimale.
        """
        if not self.is_ready:
            raise RuntimeError("Appeler .fit() avant predict_interval().")

        # 1) prédiction ponctuelle
        y_hat = self.base_rf.predict(np.asarray(x_t).reshape(1, -1))[0]

        # 2) représentation de la fenêtre courante
        X_win = self._window_residuals()

        # 3) prédictions de tous les arbres de la forêt quantile
        tree_preds = np.array([
            est.predict(X_win)[0] for est in self.rf_quant.estimators_
        ])

        # 4) balayage de β ∈ [0, α] sur une grille uniforme
        betas = np.linspace(0.0, self.alpha, self.n_grid)
        best_width = np.inf
        best_low = best_up = 0.0

        for beta in betas:
            q_low = self._rf_quantile(tree_preds, beta)
            q_up  = self._rf_quantile(tree_preds, 1.0 - self.alpha + beta)
            width = q_up - q_low
            if width < best_width:
                best_width, best_low, best_up = width, q_low, q_up

        L_t = y_hat + best_low
        U_t = y_hat + best_up
        return L_t, U_t
    
def build_X_(df_sub: pd.DataFrame, prefixes: list, static_cols: list, n: int) -> np.ndarray:
    # on garde student_id + les n premiers items
    dyn_cols = [
    col for col in df_sub.columns
    if any(col.startswith(pref) for pref in prefixes[:n])
    ]
    keep = ["email"] + static_cols + dyn_cols
    return df_sub[keep].set_index("email").values

class OneSidedSPCI_LGBM_Offline:
    """
    SPCI unilatéral [0 ; U] entraîné *hors-ligne*.

    • RandomForestRegressor  → prédiction ponctuelle f̂
    • LightGBM (quantile, alpha=1-α) → Q̂_t(1-α) calculé sur une fenêtre
      fixe de résidus (longueur w) dérivés du jeu d'apprentissage.
    """

    # ---------- initialisation ----------
    def __init__(self, alpha=0.1, w=400,
                 n_estimators=200, max_depth=-1,
                 random_state=0):

        self.alpha = alpha
        self.w     = w
        self.res_buf = deque()

        self.base_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None if max_depth == -1 else max_depth,
            random_state=random_state,
        )

        self.gbr_upper = LGBMRegressor(
            objective="quantile",
            alpha=1 - alpha,
            n_estimators=600,          # plus d’arbres
            learning_rate=0.05,        # LR plus petit
            num_leaves=15,             # plus de souplesse
            min_data_in_leaf=5,        # autoriser petits nœuds
            max_depth=5,
            random_state=random_state,
        )
        self.gbr = GradientBoostingRegressor(
            loss="quantile",
            alpha=1 - alpha,    # quantile souhaité
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=5,
            learning_rate=0.05,
            random_state=0
        )
        self.is_ready = False                 # devient True après fit()

    # ---------- entraînement offline ----------
    def fit(self, X_train, y_train):
        """Apprend f̂ et Q̂(1-α) une seule fois."""
        # 1) f̂
        self.base_rf.fit(X_train, y_train)
        print("fit 1 ok")
        # 2) résidus tronqués à 0 -> tampon
        r = np.maximum(0.0, y_train - self.base_rf.predict(X_train))
        self.res_buf.extend(r)

        # 3) fenêtre glissante -> apprentissage du quantile
        if len(self.res_buf) < self.w + 5:
            raise ValueError("Jeu d'entraînement trop court pour w={}".format(self.w))

        R = np.asarray(self.res_buf, dtype=float)
        Y = R[self.w:]                                       # cibles r_{t'}
        X = np.array([R[i - self.w:i] for i in range(self.w, len(R))])

        self.gbr.fit(X, Y)
        print("fit 2 ok")
        self.is_ready = True
        return self

    # ---------- prédiction ----------
    def predict_interval(self, x_t):
        """
        Renvoie l'intervalle [0 ; U_t] sans mise à jour.
        """
        if not self.is_ready:
            raise RuntimeError("Le modèle doit être entraîné via .fit() avant prédiction.")

        x_t = np.asarray(x_t).reshape(1, -1)
        y_hat = self.base_rf.predict(x_t)[0]

        x_feat = np.array(self._window_features()).reshape(1, -1)
        q_sup  = self.gbr.predict(x_feat)[0]
        U_t    = max(0.0, y_hat + q_sup)

        return 0.0, U_t

    # ---------- outil interne ----------
    def _window_features(self):
        """Renvoie la fenêtre de résidus utilisée à l'entraînement (longueur w)."""
        buf = list(self.res_buf)
        if len(buf) < self.w:                    # ne devrait jamais arriver après .fit()
            buf = [0.0] * (self.w - len(buf)) + buf
        return buf[-self.w:]