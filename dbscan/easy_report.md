# Density Based Spatial Clustering of Applications with Noise (DBSCAN)

DBSCAN is a clustering method that helps user classify datasets based on its spatial properties (distance).
It has several definitions that we need to understand before we discuss the actual clustering attempt.

Most clustering methods can be grouped into two categories. Partitioning and hierarchical. Partitioning basically
assigning a clusters we desired and let the algorithm minimise some function to assign each data point to one of the clusters.
On the other hand, Hierarchical algorithm create a decomposition of the data. This decomposition usually represented as
Dendogram (a graph usually for visualizing evolutionary path in biology). Each Dendogram represent a clusters. 

The nature of dendogram that it is keep splitting until each data point is its own cluster. This is not add any new information
that helps us recognize pattern in data. So we need to stop the splitting with some criteria. We let the function run recursively
and stops when meeting (or no longer meeting) some criteria.

# Notions

There are few defintions that we need to establish here. First we try to quantify the notion of clusters and noise. As a human, 
we tend to see a cluster when their grouped together and have significant distance to other grouped. Noise considered as lonely point
without any near point that can be considered as grouped together.

So now we declare a definition of neighborhood. We can declare neigborhood as

Neighborhood of point $p$ is defined by $N_{\epsilon}(p)= \{q \in D\;|\;\text{dist}(p,q) \leq \epsilon \}$ where dist is a distance function that
able toke two point
