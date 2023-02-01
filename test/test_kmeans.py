import pytest
from cluster import KMeans
from cluster import make_clusters


# Write your k-means unit tests here


# def make_clusters(
#         n: int = 500, 
#         m: int = 2, 
#         k: int = 3, 
#         bounds: tuple = (-10, 10),
#         scale: float = 1,
#         seed: int = 42) -> (np.ndarray, np.ndarray):
#     """



large_kmeans_obj=KMeans(k=20)

def test_k():
    """
    test that kmeans canbe run with values of k given
    """
   
    mat_small_n=make_clusters(n=10, k=2)
    with pytest.raises(ValueError):
        large_kmeans_obj.fit(mat_small_n[0])


    #check that k=0 is not allowed
    with pytest.raises(ValueError):
        KMeans(k=0)

        


def test_kmeans():
    """
    test something
    
    """




    pass 