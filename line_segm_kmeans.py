import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class LineSegmKMeans:
    def __init__(self, n_clusters=4, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centers_ = None
        self.labels_ = None
        self.segments_ = None
        self.centroids = None
        self.radii = None
        self.n_iter = 0

    def distance_point_to_segment(self, point, segment_start, segment_end):
        """
        Compute the distance from a point to a line segment.
        Args:
            point: numpy array of shape (n_features,)
            segment_start: numpy array of shape (n_features,)
            segment_end: numpy array of shape (n_features,)
        Returns:
            distance: float
        """
        # convert to numpy arrays
        x = np.array(point)
        segment_start = np.array(segment_start)
        segment_end = np.array(segment_end)

        seg_vec = segment_end - segment_start
        start_x = x - segment_start
        end_x = x - segment_end

        # project point onto the line defined by the segment
        start_x_dot_seg = np.dot(start_x, seg_vec)
        segvev_squared = np.dot(seg_vec, seg_vec)
        proj = start_x_dot_seg / segvev_squared

        if proj < 0:
            # closest to segment_start
            return np.linalg.norm(start_x)
        elif proj > 1:
            # closest to segment_end
            return np.linalg.norm(end_x)
        else:
            # return the perpendicular distance to the line
            segvec_norm = seg_vec / np.sqrt(segvev_squared)
            start_x_proj = segvec_norm * (start_x_dot_seg / np.sqrt(segvev_squared))
            perp_vec = start_x - start_x_proj
            return np.linalg.norm(perp_vec)
        
    def compute_central_segment(self, cluster_points, centroid):
        """
        Compute the central line segment for a set of points.
        """
        if len(cluster_points) < 2:
            return centroid, centroid, np.zeros_like(centroid)
        
        # perform pca to get direction of maximum variance
        if cluster_points.shape[1] == 1:
            direction = np.array([1.0])
        else:
            pca = PCA(n_components=1)
            pca.fit(cluster_points)
            direction = pca.components_[0]

        # Project points onto the line defined by the centroid and direction
        projections = np.dot(cluster_points - centroid, direction)
        mean__abs_proj = np.mean(np.abs(projections))
        seg_length = 2 * mean__abs_proj

        # Define segment endpoints
        segment_start = centroid - (direction * (seg_length / 2))
        segment_end = centroid + (direction * (seg_length / 2))

        return segment_start, segment_end, direction
    
    def compute_radii(self, cluster_points, segment_start, segment_end):
        """
        Compute the radius for a cluster based on distances to the line segment.
        """
        if len(cluster_points) == 0:
            return 0.0
        
        distances = []
        for point in cluster_points:
            dist = self.distance_point_to_segment(point, segment_start, segment_end)
            distances.append(dist)
        
        return np.mean(distances)


    def distance_point_to_cylinder(self, point, segment_start, segment_end, radius):
        """
        Compute the distance from a point to a line segment cylinder
        """
        dist_to_segment = self.distance_point_to_segment(point, segment_start, segment_end)
        return max(0.0, dist_to_segment - radius)


    def assign_labels(self, X, segments, radii):
        """
        Assign each point in X to the nearest line segment cluster
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))

        # distance from point to line cylinder
        for k in range(self.n_clusters):
            segment_start, segment_end = segments[k]
            radius = radii[k]
            for i in range(n_samples):
                distances[i, k] = self.distance_point_to_cylinder(X[i], segment_start, segment_end, radius)

        labels = np.argmin(distances, axis=1)
        return labels

    def fit(self, X):
        """
        Fit the Line Segment K-Means model to the data X
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        # Initialize random centroids
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        # Initialize segments and radii
        # Initially, segments are just points (centroids), and radii are zero
        self.segments_ = [(centroid, centroid) for centroid in self.centroids]
        self.radii_ = np.zeros(self.n_clusters)

        for iteration in range(self.max_iter):
            self.n_iter += 1

            # assign labels based on current segments and radii
            labels = self.assign_labels(X, self.segments_, self.radii_)

            # store previous centroids for convergence check
            prev_segments = self.segments_.copy()
            prev_radii = self.radii_.copy()

            # Update centroids, segments, and radii
            for j in range(self.n_clusters):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    # update centroid
                    self.centroids[j] = np.mean(cluster_points, axis=0)

                    # update segment
                    segment_start, segment_end, _ = self.compute_central_segment(cluster_points, self.centroids[j])
                    self.segments_[j] = (segment_start, segment_end)

                    # update radius
                    self.radii_[j] = self.compute_radii(cluster_points, segment_start, segment_end)

                else:
                    # if no points assigned, reinitialize
                    self.centroids[j] = X[np.random.randint(n_samples)]
                    self.segments_[j] = (self.centroids[j], self.centroids[j])
                    self.radii_[j] = 0.0

            # check for convergence
            converged = True
            for j in range(self.n_clusters):
                seg_start_prev, seg_end_prev = prev_segments[j]
                seg_start_curr, seg_end_curr = self.segments_[j]
                radius_prev = prev_radii[j]
                radius_curr = self.radii_[j]

                if (np.linalg.norm(seg_start_curr - seg_start_prev) > self.tol or
                    np.linalg.norm(seg_end_curr - seg_end_prev) > self.tol or
                    abs(radius_curr - radius_prev) > self.tol):
                    converged = False
                    break

            if converged:
                break

        self.labels_ = self.assign_labels(X, self.segments_, self.radii_)
        return self

def plot_clusters_2d(X, labels, segments, radii, title="Line Segment K-Means Clustering"):
    """
    Visualize 2D clustering results with cylinders
    """
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.arange(10))
    
    # Plot data points
    for i in range(len(np.unique(labels))):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                color=colors[i % 10], alpha=0.6, label=f'Cluster {i}', s=50)
    
    # Plot cylinders
    for i, ((start, end), radius) in enumerate(zip(segments, radii)):
        color = colors[i % 10]
        
        # Draw central segment
        plt.plot([start[0], end[0]], [start[1], end[1]], 
                color='black', linewidth=2, alpha=0.8)
        plt.scatter([start[0], end[0]], [start[1], end[1]], 
                color='black', s=50, marker='o', zorder=3)
        
        # Draw cylinder (rectangle with semicircles at ends)
        # Calculate direction vector and perpendicular
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length > 0:
            direction = direction / length
            # Perpendicular vector (rotated 90 degrees)
            perp = np.array([-direction[1], direction[0]])
            
    plt.title(title, fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('./images/line_segm_kmeans_on_flame.png')
    
    return plt.gcf()

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_blobs

    # Generate synthetic data
    # X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    X = pd.read_csv("../../datasets/flame.csv", index_col=False)
    X = X.drop(columns=['id', 'z'])
    X = X.to_numpy()

    # Fit LineSegmKMeans
    model = LineSegmKMeans(n_clusters=2, max_iter=100)
    model.fit(X)

    # Plot results
    plot_clusters_2d(X, model.labels_, model.segments_, model.radii_)
    plt.show()