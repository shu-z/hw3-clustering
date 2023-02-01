import pytest
from cluster import KMeans, Silhouette, make_clusters


#CAN INITIALIZE clusters with k=5 for example


#then check to see that highest silhouette score when k=5

def test_edgecase():
    pass

  


def test_optimal_k():
    """
    check that highest silhouette score is for optimal k
    """


    #make a set of clusters with k=5
    cluster_k5=make_clusters(n=500, m=12, k=5)

    #fit to kmeans

    #predict


    #get sillouette scores


    #assert that score values are between 0 and 1
    pass