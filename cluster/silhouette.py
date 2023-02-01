import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, mat: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """


   
        if mat.shape[0] != len(labels):
            raise Exception(f"Number of observations in matrix and labels are different")

        if mat.shape[0]<1:
            raise Exception(f"Matrix must have at least one observation")



        
        score_list=[]
        #get score for every observation in mat
        for i in range(0, len(mat)):
        
            #get which k cluster it's in 
            k=labels[i]
        
            #get all points in cluster, remove point i
            pts_in_cluster=np.array(np.where(labels==k))
            query_pts= pts_in_cluster[pts_in_cluster!=i]

            k_rows=mat[query_pts]
            intra_dists=cdist(k_rows, mat[[i],:], 'euclidean')
            
            #a is avg intra cluster dist 
            a=np.mean(intra_dists)
        
        
            #list of average distance to every other cluster 
            inter_clust_dist=[]
            #write this better this is dumb
            #loop through all other cluster 
            for not_k in (np.unique(labels[labels!=k])):
                not_k_rows=mat[np.where(labels==not_k)]
                #calculate dist between query pt and pts in this cluster 
                inter_dists=cdist(not_k_rows, mat[[i],:], 'euclidean')
            
                inter_clust_dist.append(np.mean(inter_dists))
            
            #b is avg intercluster dist for closest cluster 
            b=min(inter_clust_dist)          
        
            score_list.append((b-a)/max(b,a))
        
        
    
        return(score_list)  

                    
