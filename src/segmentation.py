import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

from src.features import log_transform

def scale_features(rfm):
    """Zbaton logaritem dhe pastaj StandardScaler, Kthen (tabela e shkallezuar, scaler)"""
    
    logged = log_transform(rfm)
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(logged),
        index=logged.index,
        columns=logged.columns
    )
    return scaled, scaler

def rfm_scores(rfm):
    """Piket 1-5 per R, F, M me kuantile. 5 eshte me e mira."""

    scores = pd.DataFrame(index=rfm.index)
    # Recency sa me vogel, aq me mire prandaj e nderrojm renditjen tek labels
    scores["R"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5,4,3,2,1]).astype(int)
    scores["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    scores["M"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    return scores
    
    
    
def label_segment(row):
    """Emerton segmentin sipas pikave R,F,M"""
    r, f, m = row["R"], row["F"], row["M"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f <=2:
        return "New"
    if r <= 2 and f >= 3:
        return "At risk"
    if r <= 2 and f <= 2:
        return "Lost"
    return "Needs attention"

def rfm_quantile_segments(rfm):
    """Segmentim baze me kuantile. Kthen piket R, F, M dhe kolonen Segment"""
    
    scores = rfm_scores(rfm)
    scores["Segment"] = scores.apply(label_segment, axis=1)
    return scores

def evaluate_kmeans(scaled, k_range=range(2, 11), random_state=42):
    """Per qdo k trajnon KMeans dhe kthen inertia dhe silhouette"""
    
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(scaled)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(scaled, labels)
        })
    return pd.DataFrame(rows).set_index("k")

def fit_kmeans(scaled, k, random_state=42):
    """Trajnon KMeans me k grupe dhe kthen (modeli, etiketat) me Customer ID si indeks."""
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = pd.Series(model.fit_predict(scaled), index=scaled.index, name="Cluster")
    return model, labels

def fit_hierarchical(scaled, k):
    """Trajnon Hierarchical Clustering (Ward) me k grupe dhe kthen etiketat"""
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return pd.Series(model.fit_predict(scaled), index=scaled.index, name="Cluster")

def profile_cluster(rfm, labels):
    """Profili i secilit grup: madhesia, mediana e R, F, M dhe pjesa e te ardhurave"""
    data = rfm.join(labels)
    profile = data.groupby(labels.name).agg(
        Klientet = ("Recency", "size"),
        Recency_mediana =("Recency", "median"),
        Freuqency_mediana =("Frequency", "median"),
        Monetary_mediana =("Monetary", "median"),
        Te_ardhurat =("Monetary", "sum")
    )
    
    profile["Pjesa_e_klienteve_%"] = profile["Klientet"] / profile["Klientet"].sum() * 100
    profile["Pjesa_e_te_ardhurave_%"] = profile["Te_ardhurat"] / profile["Te_ardhurat"].sum() * 100
    return profile.round(1) 

def stability_check(scaled, k, seeds=range(10)):
    """Trajnon KMeans me seeds te ndryshme dhe krahason secilin me te parin (ARI)"""
    all_labels=[
        KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(scaled)
        for s in seeds
    ]
    reference = all_labels[0]
    return [adjusted_rand_score(reference, other) for other in all_labels[1:]]