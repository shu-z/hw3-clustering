import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100, cluster_init: str = 'kmeans++'):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        if k==0:
            raise ValueError(f"k must be greater than 0")


        self.k=k
        self.tol=tol
        self.max_iter=max_iter
        self.cluster_init=cluster_init

   

    
    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        k=self.k

        if k>mat.shape[0]:
            raise ValueError(f"k is more than total number of points")

        if mat.shape[1]<2:
            raise Exception(f"this method won't run on less than 2 dimensions (sorry)")
        


        if self.cluster_init=='kmeans++':
            centroids=self._kmeans_plus_init(k, mat)
        elif self.cluster_init=='random':
            centroids=self._random_cluster_init(k, m=mat.shape[1], lower=mat.min(), upper=mat.max())
        else:
            raise Exception(f"Nonexistent cluster initialization method chosen")
        

         
        #initialize values
        sse=np.inf
        i=0
        
        old_centroids=centroids
        new_centroids=np.zeros_like(centroids)
        #while sse > tol or i<100:
        while i<self.max_iter:
                   
            #for each new set of centroids, then, calculate how far each row 
            dist_from_cent=cdist(mat, old_centroids, 'euclidean')
            #get indices for which centroid closest to each row 
            centroid_idx=(np.argmin(dist_from_cent, axis=1))
            
            #get new centroids for each cluster
            for k_idx in range(0,k):
                k_rows=np.where(centroid_idx==k_idx)
                new_centroids[k_idx]=np.mean(mat[k_rows,:], axis=1)
     
                
            #get distances and closest centroids with new centroids
            new_dist_from_cent=cdist(mat, new_centroids, 'euclidean')
            new_centroid_idx=(np.argmin(dist_from_cent, axis=1))


            #get distances of all points to closest centroids
            old_sse=np.sum((dist_from_cent[range(0, len(mat)),centroid_idx])**2)
            new_sse=np.sum((new_dist_from_cent[range(0, len(mat)),new_centroid_idx])**2)
            
            change_sse=abs(new_sse-old_sse)
            #change_sse=np.square(np.sum((old_centroids-new_centroids)**2))
            
            if change_sse<self.tol:
                self.final_centroids=new_centroids
                self.final_sse=new_sse
                return()
            
            
            else:
                old_centroids=new_centroids
                i+=1
        

        #if max iteration reached
        self.final_centroids=new_centroids
        self.final_sse=new_sse
        return()
  

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        #get distance from each point to each centroid
        dist_from_cent=cdist(mat, self.final_centroids, 'euclidean')
        #return labels by closest distance  
        return(np.argmin(dist_from_cent, axis=1))



    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if not hasattr(self, "final_sse"):
            raise AttributeError(f"No sse attribute. Try running fit")

        return(self.final_sse)

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if not hasattr(self, "final_centroids"):
            raise AttributeError(f"No centroid attribute. Try running fit")


        return(self.final_centroids)



    def _random_cluster_init(self, k, m, lower, upper):   
        """
        Initializes random centroids for k clusters and m features 
        """
        
        rand_samp=(np.random.random_sample([k,m]))
        #rescale centroids to be within range of mat values
        rand_samp_scale=(upper-lower)*rand_samp + lower

        return(rand_samp_scale)




    def _kmeans_plus_init(self, k, mat):   
        """
        Initializes centroids for k clusters and m features with kmeans++
        """
    
        centroids=[]
    
        #first, randomly select centroid for first pt 
        x=np.random.choice(range(0,mat.shape[0]), 1)
        centroids.append(mat[x])
        #mat=np.delete(mat, x, axis=0)
    
        #find remaining centroids given k
        for i in range(1, k):
            cent_list=[]
            #calculate distance between centroids and all other points
            for cent in centroids:
                dist_from_cent=cdist(mat, cent, 'euclidean')
                cent_list.append(dist_from_cent)
            
            
            #find point farthest away from all current centroids
            sum_dist=np.sum(cent_list, axis=0)
            farthest_pt_idx=np.argmax(sum_dist, axis=0)
        
            #make this point a centroid
            centroids.append(mat[farthest_pt_idx])
        
        
        return(np.concatenate(centroids))

