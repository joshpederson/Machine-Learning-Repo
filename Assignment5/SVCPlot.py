import numpy as np
import matplotlib.pyplot as plt


def plot(X, y, clf):
    plt.style.use('dark_background')

    classes = clf.classes_

    # Plot the scatter plot
    plt.scatter(X[y == classes[0], 0], X[y == classes[0], 1], s=10, c='cyan', label=classes[0])
    plt.scatter(X[y == classes[1], 0], X[y == classes[1], 1], s=10, c='yellow', label=classes[1])
    plt.subplots_adjust(right=0.8)
    plt.legend(loc=(1.04, 0))

    # Get the bounds of the plot
    ax = plt.gca()
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # Create a meshgrid of coordinates
    x_ = np.linspace(x_lim[0], x_lim[1], num=50)
    y_ = np.linspace(y_lim[0], y_lim[1], num=50)
    xx, yy = np.meshgrid(x_, y_)

    xy = np.vstack([xx.ravel(), yy.ravel()]).T

    # Create the z-coordinate for the contour plot
    z = clf.decision_function(xy).reshape(xx.shape)

    # Plot the margin
    ax.contour(xx, yy, z, levels=[-1, 0, 1], colors='w', linestyles=['--', '-', '--'])

    # Highlight the support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c='grey', alpha=0.5, s=100)

    # Display the plot
    plt.show()

def plot_surface(X, y, clf):
    plt.style.use('dark_background')

    classes = clf.classes_

    x_ = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), num=30)
    y_ = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), num=30)

    xx, yy = np.meshgrid(x_, y_)

    xy = np.vstack([xx.ravel(), yy.ravel()]).T

    # Create the z-coordinate for the contour plot
    z = clf.decision_function(xy).reshape(xx.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(xx, yy, z, alpha=0.5, color='grey')

    z_scatter = clf.decision_function(X)

    # Plot the margin
    ax.contour(xx, yy, z, levels=[-1, 0, 1], colors='black', linestyles=['--', '-', '--'], linewidths=[2, 2, 2])

    ax.scatter(X[y==classes[0], 0], X[y==classes[0], 1], z_scatter[y==classes[0]], s=10, c='cyan')
    ax.scatter(X[y == classes[1], 0], X[y == classes[1], 1], z_scatter[y == classes[1]], s=10, c='yellow')

    plt.show()
