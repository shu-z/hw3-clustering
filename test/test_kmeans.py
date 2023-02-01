import pytest
import numpy as np
from cluster import KMeans, make_clusters
from sklearn.cluster import KMeans as sklearn_kmeans


def test_k():
    """
    test that kmeans can handle wrong values of k

    """
   
   #check that k must be smaller than n
    large_km=KMeans(k=20)
    mat_small_n=make_clusters(n=10, k=2)
    with pytest.raises(ValueError):
        large_km.fit(mat_small_n[0])


    #check that k=0 is not allowed
    with pytest.raises(ValueError):
        KMeans(k=0)

        


def test_kmeans():
    """
    test kmeans fit working as expected
    
    """

    #make some clusters
    clusters=make_clusters(n=1000, k=5)    

    #fit with kmeans
    km = KMeans(k=5)
    km.fit(clusters[0])
    pred = km.predict(clusters[0])

    
    assert len(np.unique(pred))==5, "Fewer clusters output than expected"

    assert all(pred) in range(0,5), "Not all cluster labels found in expected labels"

    assert len(pred) == clusters[0].shape[0], "Length of predictions different than number of observations"
