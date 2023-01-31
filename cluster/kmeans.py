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

        self.k=k
        self.tol=tol
        self.max_iter=max_iter





    
    #check that k is not 0


    #check that k is less than total number of data points 


    #set error if Nan exists for centroid


    
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

        centroids=self.cluster_center(k, m=mat.shape[1], lower=mat.min(), upper=mat.max())
        
         
        #initialize values
        sse=np.inf
        i=0
        
        old_centroids=centroids
        new_centroids=np.zeros_like(centroids)
        #while sse > tol or i<100:
        while i<self.max_iter:
                   
            #for each new set of centroids, then, calculate how far each row 
            #print('centroids', centroids)
            dist_from_cent=cdist(mat, old_centroids, 'euclidean')
            #print('mat', mat)
            #print('dist_From_cent', dist_from_cent)
     
            #get indices for which centroid closest to each row 
            centroid_idx=(np.argmin(dist_from_cent, axis=1))
            #print(centroid_idx)
            #print('centroid_idx', centroid_idx)
            
            
            #get new centroids for each cluster
            for k_idx in range(0,k):
                k_rows=np.where(centroid_idx==k_idx)
                new_centroids[k_idx]=np.mean(mat[k_rows,:], axis=1)
     
                
            #get sse
            #this is wrong!!! need to calculate sse for clusters with old and new centroids and then take difference
            sse=np.square(np.sum((old_centroids-new_centroids)**2))
            
            if sse<self.tol:
                #print('new', new_centroids)
                self.final_centroids=new_centroids
                self.final_sse=sse
            
            
            else:
                old_centroids=new_centroids
                #print('centroids', centroids)
                i+=1
  

    def predict(self, mat: np.ndarray) -> np.ndarray:

        dist_from_cent=cdist(mat, self.final_centroids, 'euclidean')

        return(np.argmin(dist_from_cent, axis=1))
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

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return(self.final_sse)

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        return(self.final_centroids)



    def cluster_center(k, m, lower, upper):   
        """
        Initializes random centroids for k clusters and m features 
        """
        rand_samp=(np.random.random_sample([k,m]))
        #rescale random sample so that they are within bounds of all values of mat
        rand_samp_scale=(upper-lower)*rand_samp + lower
        return(rand_samp_scale)




    def kmeans_plus(k, mat):   
        """
        Initializes centroids for k clusters and m features with kmeans++
        """
    
        centroids=[]
    
        #first, randomly select centroid for first pt 
        x=np.random.choice(range(0,mat.shape[0]), 1)
        centroids.append(mat[x])
    
    
        #mat=np.delete(mat, x, axis=0)
    
    
        #calculate distance between centroid and all other points
    
        for i in range(1, k):
        
            cent_list=[]
            for cent in centroids:
                dist_from_cent=cdist(mat, cent, 'euclidean')
                cent_list.append(dist_from_cent)
            
        
            sum_dist=np.sum(cent_list, axis=0)
        
            farthest_pt_idx=np.argmax(sum_dist, axis=0)
        
        
            centroids.append(mat[farthest_pt_idx])
        
        
        return(np.concatenate(centroids))

