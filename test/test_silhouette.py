import pytest
import numpy as np
from cluster import KMeans, Silhouette, make_clusters
from sklearn.metrics import silhouette_score



def test_silhouette():
    """
    check that silhouette scores accurate against sklearn
    """


    #make a set of clusters
    clusters=make_clusters(n=500, m=12, k=4)

    #fit with kmeans, and predict
    km = KMeans(k=4)
    km.fit(clusters[0])
    pred = km.predict(clusters[0])

    #get sillouette scores
    scores = Silhouette().score(clusters[0], pred)

    #assert that score values are between 0 and 1
    assert all( (x >= -1 and x<=1) for x in scores), "Silhouette scores outside of range from -1 to 1"


    #get silhouette scores from sklearn
    sklearn_scores=silhouette_score(clusters[0], pred, random_state=42)
    #assert that scores match up
    assert np.isclose(np.mean(scores), sklearn_scores), "These silhouette scores differ from sklearn scores"
    