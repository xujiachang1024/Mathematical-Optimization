# import dependencies
import numpy as np
import matplotlib.pyplot as plt


# load dataset from text file
def load_dataset(file):
    return np.loadtxt(fname=file)


# calculate Euclidian distance between 2 data points
def euclidian(a, b):
    return np.linalg.norm(a - b)


# k-mean clustering algorithm
def kmeans(dataset, k, episilon=0, distance="euclidian", max_iter=float('inf')):

    # list to store past centroids
    history_centroids = []

    # set distance calculation method
    if distance == "euclidian":
        dist_method = euclidian

    # find the dimension of the dataset
    (num_instances, num_features) = dataset.shape

    # initialize centroids at random locations
    new_centroids = dataset[np.random.randint(0, num_instances - 1, size=k)]
    history_centroids.append(new_centroids)
    old_centroids = np.zeros(new_centroids.shape)

    # store clusters
    belongs_to = np.zeros((num_instances, 1))

    displacement = dist_method(new_centroids, old_centroids)
    iteration = 0

    # repeat until convergence
    while displacement > episilon and iteration < max_iter:

        # for each instance in the dataset
        for index_instance, instance in enumerate(dataset):
            distance_vector = np.zeros((k, 1))

            # for each centroid (k of them)
            for index_centroid, centroid in enumerate(new_centroids):
                distance_vector[index_centroid] = dist_method(instance, centroid)

            # find the closest centroid, and assign the instance to the cluster
            belongs_to[index_instance, 0] = np.argmin(distance_vector)

        tmp_centroids = np.zeros((k, num_features))

        # for each centoid (k of them)
        for index in range(new_centroids.shape[0]):

            # find all the instances that belong to this cluster
            close_instances = [i for i in range(len(belongs_to)) if belongs_to[i] == index]

            # find the mean of these close instances, which will be the new centroid
            tmp_centroids[index, :] = np.mean(dataset[close_instances], axis=0)

        # update centroids cache
        old_centroids = new_centroids
        new_centroids = tmp_centroids
        history_centroids.append(new_centroids)

        displacement = dist_method(new_centroids, old_centroids)
        iteration += 1

    return new_centroids, history_centroids, belongs_to


def plot(dataset, history_centroids, belongs_to):
    # we'll have 2 colors for each centroid cluster
    colors = ['r', 'g']

    # split our graph by its axis and actual plot
    fig, ax = plt.subplots()

    # for each point in our dataset
    for index in range(dataset.shape[0]):
        # get all the points assigned to a cluster
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        # assign each datapoint in that cluster a color and plot it
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))

    # let's also log the history of centroids calculated via training
    history_points = []
    # for each centroid ever calculated
    for index, centroids in enumerate(history_centroids):
        # print them all out
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                print("centroids {} {}".format(index, item))

    plt.show()


def main():
    dataset = load_dataset('durudataset.txt')
    centroids, history_centroids, belongs_to = kmeans(dataset=dataset, k=2)
    plot(dataset, history_centroids, belongs_to)


main()
