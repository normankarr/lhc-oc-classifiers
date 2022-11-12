import numpy as np
from sklearn.svm import OneClassSVM

class IAD(OneClassSVM):
    """
    An inner anomaly detector will build a hyperplane around the interior of the distribution
    """
    def __init__(self, kernel="rbf", gamma=0.1, nu=0.85):
        """
        nu is set to 0.9 because to build a hyperplane along the interior, we want to force all of the data outside of the hyperplane by classifying it as -1 instead of 1
        """
        super().__init__(kernel=kernel, gamma=gamma, nu=nu)
        
    def predict(self, X):
        """
        predict 0 if not an outlier
        predict 1 if outlier
        """
        return (super().predict(X)+1)//2
    
    def scores(self, X):
        return 1-super().score_samples(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred==y)/len(y)
    
class OAD(OneClassSVM):
    """
    An outer anomaly detector will build a hyperplane around the exterior of the distribution
    """
    def __init__(self, kernel="rbf", gamma=0.1, nu=0.15):
        """
        nu is set to 0.1 because to build a hyperplane along the exterior, we want to force all of the data inside of the hyperplane by classifying it as 1
        """
        super().__init__(kernel=kernel, gamma=gamma, nu=nu)
        
    def predict(self, X):
        """
        predict 0 if not an outlier
        predict 1 if outlier
        """
        return (1-super().predict(X))//2
    
    def scores(self, X):
        return  super().score_samples(X)
        
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred==y)/len(y)