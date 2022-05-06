# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 12:21:59 2021

@author: johan
"""




import numpy as np
import random
import scipy.stats as ss
import matplotlib.pyplot as plt
from collections import Counter
    
def distance(p1, p2):
    """
    

    Parameters
    ----------
    p1 : NumPy array dtype=int
        POINT 1
    p2 : NumPy array dtype=int
        POINT 2

    Returns
    -------
    int value
        DISTANCE BETWEEN THE TWO POINTS.

    """
    return np.sqrt(np.sum(np.power(p2-p1,2)))
    

def majority_vote(votes):
    """
    Returns the most repeated element, if tie: random
    """

    count_votes=Counter(votes)
    winners=[]
    max_count = max(count_votes.values())
    for vote,count in count_votes.items():
        if count == max_count:
            winners.append(vote)
    return random.choice(winners)        
    #max_key = max(count_votes, key = lambda x:count_votes[x] )



def majority_vote_short(votes):
    """
    Shorter and faster way to find most recurring element
    Random selection is not used
    """
    mode, count = ss.mstats.mode(votes)
    return mode

def find_nearest_neighbors(p, points, k=5):
    """
    Find k nearest neighbours and return their indices
    """

    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p , points[i])
    ind = np.argsort(distances)
    return ind[:k]    



def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind])


def generate_synth_data(n=50):
    """
    generates n data for 2 different sets of points from bivariate normal distributions with mean 0 and 1
    """
    points = np.concatenate((ss.norm(0,1).rvs((n,2)),ss.norm(1,1).rvs((n,2))), axis=0)
    outcomes = np.concatenate((np.repeat(0,n),np.repeat(1,n)))
    return (points, outcomes)                         




def make_prediction_grid(predictors, outcomes, limits, h ,k):
    """Classify each point on the prediction grid"""
    
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    
    prediction_grid = np.zeros(xx.shape, dtype=int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i]=knn_predict(p, predictors, outcomes, k)
    return (xx, yy, prediction_grid)



def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue","yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5, shading='auto')
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    #plt.savefig(filename)
    
from sklearn import datasets
iris = datasets.load_iris()

predictors = iris.data[:,0:2]
outcomes = iris.target
plt.plot(predictors[outcomes==0][:,0],predictors[outcomes==0][:,1],"ro")
plt.plot(predictors[outcomes==1][:,0],predictors[outcomes==1][:,1], "go")
plt.plot(predictors[outcomes==2][:,0],predictors[outcomes==2][:,1],"bo")    
    
"""
(predictors, outcomes) = generate_synth_data()

k=5; limits= (-3,4,-3,4); h=0.1
#value of k must be optimised -  bias-variance tradeoff
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid (xx, yy, prediction_grid, 'filename')

"""
k=5; limits= (4,8, 1.5,4.5); h=0.1
#value of k must be optimised -  bias-variance tradeoff
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
print(prediction_grid)
plot_prediction_grid (xx, yy, prediction_grid, 'filename')


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)


my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors])
my_predictions == sk_predictions

print(100 * np.mean(sk_predictions == my_predictions))
print(100 * np.mean(sk_predictions == outcomes))
print(100 * np.mean(my_predictions == outcomes))
