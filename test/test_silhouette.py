import pytest
from cluster import KMeans, Silhouette


#CAN INITIALIZE clusters with k=5 for example


#then check to see that highest silhouette score when k=5

def test_optimal_k():
    """
    check that highest silhouette score is for optimal k
    """