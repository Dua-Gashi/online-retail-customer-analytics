import pandas as pd
from sklearn.preprocessing import StandardScaler

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
