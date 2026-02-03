import copy
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

def flatten(grads):
    return torch.cat([g.view(-1) for g in grads])

def unflatten(flat_grad, ref_grads):
    recovered = []
    offset = 0
    for g in ref_grads:
        numel = g.numel()
        recovered.append(flat_grad[offset:offset+numel].view(g.shape))
        offset += numel
    return recovered

def get_mean_and_std(gradient_list):
    # Stack the list of 1D tensors: (num_clients, num_params)
    stacked_grads = torch.stack(gradient_list)
    return torch.mean(stacked_grads, dim=0), torch.std(stacked_grads, dim=0)

def get_loss_n_accuracy(model, criterion, data_loader, args, num_classes=10):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    
    # disable BN stats during inference
    model.eval()                                      
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
            
    # forward-pass to get loss and predictions of the current batch
    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                labels.to(device=args.device, non_blocking=True)
                                            
        # compute the total loss over minibatch
        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)
        total_loss += avg_minibatch_loss.item()*outputs.shape[0]
                        
        # get num of correctly predicted inputs in the current batch
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        # fill confusion_matrix
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
                                
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)

import matplotlib.pyplot as plt
import seaborn as sns

class DBHC:
    """
    DBHC: A DBSCAN-based hierarchical clustering algorithm.
    Based on the paper by Alireza Latifi-Pakdehi and Negin Daneshpour (2021).
    Extended to support automatic cluster determination (ULDBHC) and configurable metrics.
    """

    def __init__(self, n_clusters=None, metric='manhattan', plot_merge_cost=False):
        """
        Args:
            n_clusters (int or None): The desired number of clusters (k).
                                      If None, the number of clusters is determined automatically
                                      using an elbow heuristic on merge distances.
            metric (str): The distance metric to use. 
                          Options: 'manhattan' (default), 'euclidean', or any metric supported by sklearn.
            plot_merge_cost (bool): If True, plots the merge distance history to visualize the "jump".
        """
        self.n_clusters = n_clusters
        self.metric = metric
        self.labels_ = None
        self.cluster_centers_ = None
        self.plot_merge_cost = plot_merge_cost

    def fit(self, X):
        """
        Perform clustering on X.

        Args:
            X (array-like): shape (n_samples, n_features)
        
        Returns:
            self
        """
        X = np.array(X)
        m = X.shape[0] # Number of data objects
        
        # --- Step 1: Generating value of Eps Parameter (Fig. 3) ---
        # "Dist(i) = distance from object i to its 2th nearest neighbor"
        # The paper says: "When MinPts is set to 3, we have to look for the second nearest neighbor."
        
        # We find k=2 nearest neighbors. The first one is the point itself (dist=0), 
        # the second one is the actual nearest neighbor.
        nbrs = NearestNeighbors(n_neighbors=2, metric=self.metric).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Dist stores the distance to the 2nd nearest neighbor (index 1)
        dist_vec = distances[:, 1]
        
        # sort Dist in Ascending manner
        dist_vec_sorted = np.sort(dist_vec)
        
        E = []
        step_size = int(np.sqrt(m))
        j = step_size
        
        # Select Eps values at regular intervals
        while j < m:
            E.append(dist_vec_sorted[j])
            j += step_size
            
        # Ensure E is sorted (it should be, as dist_vec_sorted is sorted)
        E = sorted(list(set(E))) # Remove duplicates just in case, though logic implies distinct if values distinct
        
        # --- Step 2: Identifying initial clusters (Fig. 4) ---
        # We need to track which points are assigned to which cluster.
        # Initialize all labels to -1 (noise)
        final_labels = np.full(m, -1)
        
        # Keep track of unclustered points indices
        remaining_indices = np.arange(m)
        current_cluster_id = 0
        
        # E is sorted ascending.
        # We iterate through Eps values.
        
        for eps in E:
            if eps <= 1e-6:  # Avoid too small Eps
                eps = 1e-6

            if len(remaining_indices) == 0:
                break
                
            # Prepare data subset
            X_subset = X[remaining_indices]
            
            if X_subset.shape[0] == 0:
                break

            # Execute DBSCAN(D, e, 3)
            # MinPts is fixed to 3 as per paper Section 3.1
            db = DBSCAN(eps=eps, min_samples=3, metric=self.metric)
            db.fit(X_subset)
            
            subset_labels = db.labels_
            
            # Identify which points in subset formed clusters (label != -1)
            # We need to assign unique cluster IDs.
            unique_subset_labels = set(subset_labels)
            
            mask_clustered_in_subset = (subset_labels != -1)
            
            if not np.any(mask_clustered_in_subset):
                continue
            
            # For each cluster found in this step
            for label in unique_subset_labels:
                if label == -1:
                    continue
                
                # Get indices in X_subset belonging to this cluster
                cluster_indices_in_subset = np.where(subset_labels == label)[0]
                
                # Map back to original indices
                original_indices = remaining_indices[cluster_indices_in_subset]
                
                # Assign unique global cluster ID
                final_labels[original_indices] = current_cluster_id
                current_cluster_id += 1
            
            # Remove clustered points from remaining_indices
            # The indices that were clustered in this round:
            clustered_indices_subset = np.where(subset_labels != -1)[0]
            remaining_indices = np.delete(remaining_indices, clustered_indices_subset)
            
        # Handle points that remained noise after all Eps
        
        unique_labels = sorted(list(set(final_labels)))
        if -1 in unique_labels:
            unique_labels.remove(-1)
            
        clusters = {}
        for uid in unique_labels:
            clusters[uid] = np.where(final_labels == uid)[0]
            
        # --- Step 3: Merging the initial clusters (Fig. 5) ---
        
        # Helper to calculate centroid
        def get_centroid(indices):
            return np.mean(X[indices], axis=0)
            
        cluster_centroids = {uid: get_centroid(indices) for uid, indices in clusters.items()}
        
        # Keep initial state for auto-k replay
        initial_clusters = copy.deepcopy(clusters)
        initial_centroids = copy.deepcopy(cluster_centroids)
        
        current_k = len(clusters)
        target_k = self.n_clusters if self.n_clusters is not None else 1
        
        merge_history = [] # Stores (dist, c1, c2)
        
        # If we have noise points (label -1), should they be merged?
        # The paper doesn't strictly specify handling of final noise.
        # We will proceed merging the identified valid clusters.
        
        while current_k > target_k:
            # Find two cluster C and C' with nearest center
            min_dist = np.inf
            
            uids = list(clusters.keys())
            
            # Calculate pairwise distances between centroids
            # This is O(N^2) relative to number of clusters.
            
            # Get all centroids as array to use efficient cdist
            centroid_ids = uids
            centroid_matrix = np.array([cluster_centroids[uid] for uid in centroid_ids])
            
            if len(centroid_matrix) < 2:
                break
                
            dists = pairwise_distances(centroid_matrix, centroid_matrix, metric=self.metric)
            
            # Set diagonal to infinity to ignore self-distance
            np.fill_diagonal(dists, np.inf)
            
            # Find minimum
            flat_idx = np.argmin(dists)
            row_idx, col_idx = np.unravel_index(flat_idx, dists.shape)
            
            min_dist = dists[row_idx, col_idx]
            
            c1_id = centroid_ids[row_idx]
            c2_id = centroid_ids[col_idx]
            
            # Record merge intent
            merge_history.append({'dist': min_dist, 'c1': c1_id, 'c2': c2_id})

            # Merge C and C'
            # Create new ID (re-use c1_id, remove c2_id)
            new_indices = np.concatenate([clusters[c1_id], clusters[c2_id]])
            clusters[c1_id] = new_indices
            del clusters[c2_id]
            
            # Calculate centroid of new created cluster
            cluster_centroids[c1_id] = get_centroid(new_indices)
            del cluster_centroids[c2_id]
            
            current_k -= 1
        
        # If auto mode, determine best k and replay
        if self.n_clusters is None and merge_history:
            dists = [rec['dist'] for rec in merge_history]
            
            # Heuristic: Find the largest jump in merge distances.
            # We want to stop BEFORE the jump.
            stop_idx = 0
            diffs = [] # Initialize diffs for plotting scope

            if len(dists) > 0:
                if len(dists) == 1:
                    # Only one merge possible. If we do it, k=1. If not, k=2.
                    # Default to k=2 (no merge) for better separation usually.
                    stop_idx = 0
                else:
                    diffs = np.diff(dists)
                    if len(diffs) > 0:
                        # Find the index where the distance jumps the most
                        # diff[i] = dist[i+1] - dist[i]
                        # A large jump at i means merging i+1 is much more expensive than i.
                        # So we perform merges 0..i, and stop before i+1.
                        # stop_idx = i + 1
                        max_diff_idx = np.argmax(diffs)
                        stop_idx = max_diff_idx + 1
            
            # --- Visualization Block ---
            if self.plot_merge_cost and len(dists) > 1:
                try:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot Merge Distances (Blue)
                    plt.plot(range(len(dists)), dists, marker='o', label='Merge Distance', color='blue')
                    
                    # Plot Diffs (Red dashed)
                    if len(diffs) > 0:
                        # Pad diffs to align with dists (diff[i] is between i and i+1)
                        # We plot it at i+1 to signify the jump TO that step
                        plt.plot(range(1, len(dists)), diffs, marker='x', linestyle='--', label='Jump (Diff)', color='red')
                    
                    # Mark the Cutoff
                    if len(diffs) > 0:
                        plt.axvline(x=stop_idx, color='green', linestyle=':', linewidth=2, label=f'Cutoff (idx={stop_idx})')
                        
                    plt.title("DBHC Merge Cost Analysis (Auto-k)")
                    plt.xlabel("Merge Step Index")
                    plt.ylabel("Distance / Cost")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.savefig("dbhc_merge_cost.png")
                    plt.close()
                    print("Saved merge cost visualization to 'dbhc_merge_cost.png'")
                except Exception as e:
                    print(f"Failed to plot merge cost: {e}")
            # ---------------------------

            # Restore and Replay
            clusters = initial_clusters
            cluster_centroids = initial_centroids
            
            for i in range(stop_idx):
                rec = merge_history[i]
                c1 = rec['c1']
                c2 = rec['c2']
                
                new_indices = np.concatenate([clusters[c1], clusters[c2]])
                clusters[c1] = new_indices
                del clusters[c2]
                
                cluster_centroids[c1] = get_centroid(new_indices)
                del cluster_centroids[c2]
            
        # Update labels based on merged clusters
        new_final_labels = np.full(m, -1)
        
        # Remap arbitrary cluster IDs to 0..k-1 range
        final_mapping = {}
        for idx, (old_id, indices) in enumerate(clusters.items()):
            new_final_labels[indices] = idx
            final_mapping[old_id] = idx
            
        self.labels_ = new_final_labels
        self.cluster_centers_ = np.array([cluster_centroids[uid] for uid in clusters.keys()])
        
        return self
