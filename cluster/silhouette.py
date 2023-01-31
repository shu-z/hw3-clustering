import numpy as np
from scipy.spatial.distance import cdist


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

    
        score_list=[]
        for i in range(0, len(mat)):
        
        
            #get which k it's in 
            k=labels[i]
        
            #get all other points in cluster
            k_rows=mat[np.where(labels==k)]
            intra_dists=cdist(k_rows, mat[[i],:], 'euclidean')
            
            #a is avg intra cluster variance 
            a=np.mean(intra_dists)
        
        
            #b is intercluster variance
 
        
        
            inter_clust_dist=[]
            #write this better this is dumb
            for not_k in (np.unique(labels[labels!=k])):
                #print('not_k', not_k)
                not_k_rows=mat[np.where(labels==not_k)]
                inter_dists=cdist(not_k_rows, mat[[i],:], 'euclidean')
            
                inter_clust_dist.append(np.mean(inter_dists))
            
        
            b=min(inter_clust_dist)          
        
            score_list.append((b-a)/max(b,a))
        
        
    
        return(score_list)  

                    
