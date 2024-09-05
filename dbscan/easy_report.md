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

> [!IMPORTANT]
> Definition 1: The Epsilon neighborhood $\epsilon$ of a point $p$. denoted by $N_{\epsilon}(p)$, is defined by $N_{\epsilon}(p)= \{q \in D\;|\;\text{dist}(p,q) \leq \epsilon \}$

![definition 1](/assets/images/dbscan_definition_1.png)

So let's say we define our distance using euclidean distance. We declare that our epsilon neighbor vaule is 5. Assume we pick point $p$ that located in (0,0) coordinate.
Any point that inside the radius of 5 would be considered as epsilon neighbor of $p$. We notate it as $N_{\epsilon}(p)$ and it would have set of points. As you may infer,
a point can be neighbor to each other.

This approach can helps us establish the concept of cluster but not enough since a line of point that each of the line is the epsilon neighbor of each
other can be considered as cluster which something that we dont want. We introduce the second concept called minumum point.

> [!IMPORTANT]
> Definition 2: Directly density reachable. A point $p$ is *directly density reachable* from a point $q$ with respect to epsilon  if
> 	1. $p \in N_\epsilon (q)$
> 	2. $| N_\epsilon(q)| \geq P_{\text{min}}$

![definition 2](/assets/images/dbscan_definition_2.png)

We define concept directly densiry reachable which use previous epsilon neighbor notation and add new rules which the total epsilon neighbor must
exceed or equal certain poisitive integer to be considered directly density reachable. 
Symmetry in directly density reachable in only for *core points* but not for *edge points*. 
This is because core points can always be fulfill the minimum point rules whereas edge point are not because they are in the edge of clusters.
Using this concept, we can declare the third definition.

> [!IMPORTANT]
> Definition 3: Density reachable, A point $p$ is density reachable from a point $q$ with respect to epsilon neighbors and minimum points. if there is a chain of points $p_1, \cdots, p_n$ where $p_1= q$ and $p_n = p$ such that $p_{i+1}$ is directly density reachable from $p_i \in p_1,\cdots,p_n$

![definition 3](/assets/images/dbscan_definition_3.png)

We add new definition so the edge point of clusters can be linked with core points. 
The relation is transitive meaning the reachability relataion can be extended through other point but not symmetric.
The core point can reach the edge point through series of point but edge point cannot because it fail at minimum point rule.
So how do we know whether two edge points belong to a same cluster? Enter definition 4

> [!IMPORTANT]
> Definition 4: Density connected. A point $p$ is density connected to a point $q$ with respect to epsilon neighbors and minimum points if there is a point $o$ such that both $p$ and $q$ are density reachable from $o$ with respect to epsilon neighbors and minimum point.

![definition 4](/assets/images/dbscan_definition_4.png)

So if two edge points is density connected to each other then we could assign both to the same cluster. 
With this four definition, we can declare our cluster and noise defintion.

> [!IMPORTANT]
> Definition 5: Cluster. Let $D$ be a database of points. A cluster $C$ with respect to epsilon neighbors and minimum points is a non empty subset of $D$ satisfying this following condition
> 1. $\forall p, q:$ if $p \in C$ and $q$ is  density reachable for $p$ with respect to epsilon neighbors an minimal points, then $q \in C$. This would called as Maximality
> 2. $\forall p, q \in C: p$ is a density connected to $q$ with respect to epsilon neighbors and minimal points. This would be called as Connectivity 

![definition 5](/assets/images/dbscan_definition_5.png)

> [!IMPORTANT]
> Definition 6: Noise, Let $C_1, \dots, C_k$ be the clusters of the database $D$ with respect to parameters $\epsilon_i$ and $P_{min}^i$ and $i = 1, \dots, k$ then we define noise as the set of points in the database $D$ not belonging to any cluster $C_i$. $\{p \in D\;|\;\forall i: p \notin C_i\}$

![definition 6](/assets/images/dbscan_definition_6.png)

Point that included in maximality is the core point and the one that included in connectivity is the edge point. Connectivity is the superset of maximality.
Noise, on the other hand, does not belong to any cluster since it does not meet any definition we define earlier. 
Intuitively, we can start clustering by picking random point and check whether that point fulfill the definition 2, 
if yes that we can recursively find the edge of cluster until all point is visited. Then we can continue picking new non visited random point and
start clustering again. We do this until all points are calculated. All points that does not belong to any cluster can be considered as noise.

# Application

One thing to note about the defintion 6 is that we consider that each cluster have their own minimum point $P_{min}$ and epsilon neighbor $N_{\epsilon}(p)$.
In the actual DBSCAN application, we define both minumum point and epsilon neighbor as constant and applied to all clusters.

As we discuss earlier, DBSCAN can start with random point. Roughly we can create the pseudocode like this

```python
def dbscan(dataset, eps, min_point):
    visited = []
    while not is_all_point_visited(visited, points):
        point = pick_random_point(dataset, visited)
        cluster_count = get_cluster_count()
        if cluster_recursive(point, cluster_count, visited):
            increment_cluster_count()
    return cluster_count

def cluster_recursive(point, cluster_no, visited):
    if not is_core_point(point) and is_clusterless(point):
        mark_as_noise(neighbor) 
        set_visited(point)
        return False
    
    assign_cluster(point, cluster_no)
    neighbor = get_epsilon_neighbor()
    for n in neighbor:
        if n in visited:
            continue
        if is_connected(point, n):
            assign_cluster(neighbor, cluster_no)
        set_visited(n)
        cluster_recursive(point, cluster_no)
    return True
```

This is a long pseudocode but give roughly how DBSCAN works. First we have empty container `visited` which will help us to mark which point
we already evaluate. We do this to avoid endless loop becase some points might be a neighbor of each other.
We check whether we already visit all the points. This is our termination condition. We need this because we use recursive function.
We pick random point that are not visited yet then we evalute it. 

Function `cluster_recursive` evaluate a point, add it to visited. In the evaluation, we check whether the point is a core point
which means it contains some minimum point count inside its epsilon neighbors. If it is not then we mark it as a noise; not belong to any cluster.
If it contains minumum point, then we loop to all of its neighbor. If we already visit it, then we will skip it.
If it is connected (see definition 5), then we can mark it as same clusters as point. Then for all its neighbor, we repeat the process until all neighbors is already visited.
The few noise point will get visited via random pick point--it wont pick already visited point.
When we already visit all point, then we can stop the iteration and return the clustering we done.

## Epsilon and Minimal Point Parameters

In the psudocode above, we see that we need to input epsilon and minimal point.
One may ask what is the best epsilon and minimal point to get the most of DBSCAN?
The DBSCAN original paper use something called *sorted K-dist graph*.

Let's assume we pick a random point in a dataset.
We can calculate the distance between this point and the rest of the dataset.
This distance we can sort for the furthest to the nearest. This what *sorted K-dist graph*
K-means distance mean the distance between point.
K can be switched with integer. It represent the order of point in distance notation.
For example, the nearest point would be the 1-dist, second nearest point would be 2-dist and so on.

![sorted k dist](/assets/images/dbscan_sorted_k_dist.png)

We sort it in from the furthest point to nearest point. We assume that furthest point are noise (at least that what the paper assumes).
The line would get flatten because it reach a cluster but not the reference point cluster. The minumum point, recommended by the paper, is
to find the first flatten line when ordering k-distance from the furthest to the nearest. The paper found that 4 is the acceptable number.
Note that this is tested in two dimensional data. This is still relay on a user see the sorted K-dist graph because there is a chance that
we pick a reference point that turns out to be a noise.

![sorted k dist valley](/assets/images/dbscan_sorted_k_dist_valley.png)

What about the epsilon distance? Using same sorted K-dist graph, we can find a better epsilon distance.
The difference is that the graph are sorted from the nearest to the furthest.
The epsilon can be the first "elbow" that we encounter.
Some discussion points out that some cluster might be smaller than our reference point cluster thus
we need to reduce it by half to accomodate smaller cluster.

![sorted k dist elbow](/assets/images/dbscan_sorted_k_dist_elbow.png)

# Conclusion

DBSCAN is a clustering algorithm that uses notion of distance between point for deciding whether a point belong to same cluster are not.
The algorithm itself are recursive thus require a termination condition. The termination is when all dataset is visited by the function.
Intuitively, DBSCAN works by checking whether a point is in a distance (epsilon), it if where then it possibly belong to cluster.
To avoid having many cluster, DBSCAN introduce a notion of minimum number of neighbor before consider it as a one cluster.
DBSCAN excel in detecting a cluster as long as it has considrable distance among it. It would fail if the cluster does not have
clear notion of distance. 

# Reference
- [Ester, Martin. (1996). A Density Based Algorithm for Discovering Clusters in Large Spatial Databases With Noise. Proceedings in 2nd International Conference of Knowledge Discovery and Data Mining. KDD-96](https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf)













