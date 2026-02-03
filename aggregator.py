import copy
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.cluster import DBSCAN, OPTICS
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sklearn.metrics.pairwise as smp
from scipy.spatial.distance import cityblock, cdist, cosine
from collections import Counter
from sklearn.decomposition import PCA
from utils import DBHC

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

def apply_pcgrad_for_two_grads(g_i_list, g_j_list):
    """
    如果 g_i 与 g_j 冲突，则将 g_i 投影到 g_j 的法平面上。
    公式: g_i = g_i - ( (g_i · g_j) / ||g_j||^2 ) * g_j
    
    Args:
        g_i_list: list of Tensors (待处理的梯度/更新)
        g_j_list: list of Tensors (参考的梯度/更新)
        
    Returns:
        list of Tensors: 处理冲突后的 g_i
    """
    # 1. 展平为向量以进行点积运算
    # 使用 flatten 并 cat 确保所有维度的参数都被考虑到
    g_i_flat = torch.cat([p.view(-1) for p in g_i_list])
    g_j_flat = torch.cat([p.view(-1) for p in g_j_list])
    
    # 2. 计算点积检查冲突
    dot_product = torch.dot(g_i_flat, g_j_flat)
    
    # 3. 如果点积为负，执行投影
    if dot_product < 0:
        # 计算 g_j 的模平方
        norm_sq_j = torch.dot(g_j_flat, g_j_flat) + 1e-9
        
        # 计算投影后的扁平向量
        # g_i_new = g_i - (dot / norm_sq) * g_j
        g_i_flat_new = g_i_flat - (dot_product / norm_sq_j) * g_j_flat
        
        # 4. 将扁平向量恢复为原始的 List of Tensors 结构
        new_g_i_list = []
        start_idx = 0
        for original_p in g_i_list:
            numel = original_p.numel()
            # 裁剪并重塑形状
            new_p = g_i_flat_new[start_idx : start_idx + numel].view(original_p.shape)
            new_g_i_list.append(new_p)
            start_idx += numel
            
        return new_g_i_list
    
    # 如果没有冲突，直接返回原始的 g_i (或者它的副本)
    return [p.clone() for p in g_i_list]

def aggregate_fedavg(client_updates, **kwargs):
    """FedAvg"""
    num_clients = len(client_updates)
    total_update = client_updates[0]
    # 5A. 遍历剩余客户端，进行累加
    for i in range(1, len(client_updates)):
        total_update += client_updates[i]        
    # 6A. 计算平均值
    update_avg = total_update / num_clients

    return update_avg

# todo: use weight 
def aggregate_krum(client_updates, args, **kwargs):
    """Krum"""
    f = int(args.num_clients * args.mal_prop)
    n = len(client_updates)
    if n <= 2 * f + 2: 
        # Fallback
        return aggregate_fedavg(client_updates)
        
    flat_updates = torch.stack([flatten(g) for g in client_updates])
    scores = []
    # calculate the dist matrix
    dists = torch.cdist(flat_updates, flat_updates, p=2)
    
    for i in range(n):
        # get the top k nearest neighbor
        k_nearest = dists[i].topk(n - f - 1, largest=False).values
        scores.append(k_nearest.sum())
        
    best_idx = torch.argmin(torch.tensor(scores))
    return client_updates[best_idx]

def aggregate_fltrust(client_updates, server_update, **kwargs):
    """FLTrust"""
    flat_server = flatten(server_update)
    norm_server = torch.norm(flat_server)
    
    weighted_sum = torch.zeros_like(flat_server)
    total_weight = 0.0
    
    for updates in client_updates:
        flat_client = flatten(updates)
        norm_client = torch.norm(flat_client)
        cos_sim = F.cosine_similarity(flat_client, flat_server, dim=0)
        relu_sim = F.relu(cos_sim)
        
        normalized_client = (norm_server / (norm_client + 1e-9)) * flat_client
        
        weighted_sum += relu_sim * normalized_client
        total_weight += relu_sim
        
    if total_weight > 1e-6:
        final_flat = weighted_sum / total_weight
    else:
        final_flat = flat_server
        
    return unflatten(final_flat, server_update)

def aggregate_fltrust2(client_updates, server_update, **kwargs):
    """FLTrust"""
    norm_server = torch.norm(server_update)
    client_updates_list = []
    for i, update in enumerate(client_updates):
        update_np = update.cpu().numpy()
        if np.isnan(update_np).any():
                print(f"Client {i} has NaN in updates! Skipping.")
        else:
            client_updates_list.append(update)
        
    if not client_updates_list:
        return server_update
    
    weighted_sum = torch.zeros_like(server_update)
    total_weight = 0.0
    
    for updates in client_updates_list:
        norm_client = torch.norm(updates)
        cos_sim = F.cosine_similarity(updates, server_update, dim=0)
        relu_sim = F.relu(cos_sim)
        
        normalized_client = (norm_server / (norm_client + 1e-9)) * updates
        
        weighted_sum += relu_sim * normalized_client
        total_weight += relu_sim
        
    if total_weight > 1e-6:
        final_update = weighted_sum / total_weight
    else:
        final_update = server_update

    return final_update

def aggregate_priroagg_rfa(client_grads, args, **kwargs):
    """
    PriRoAgg: Achieving Robust Model Aggregation With Minimum Privacy Leakage for Federated Learning
    RFA: Robust Aggregation for Federated Learning
    """
    nu=1e-6
    device = client_grads[0][0].device
    num_clients = len(client_grads)
    alphas = args.rfa_alphas

    # Step 1: flatten all client gradients
    flat_clients_tensor = torch.stack([flatten(g) for g in client_grads])

    D = flat_clients_tensor[0].numel()

    # weights
    if alphas is None:
        alphas = torch.ones(num_clients, dtype=torch.float64, device=device) / num_clients
    else:
        alphas = torch.tensor(alphas, dtype=torch.float64, device=device)
        alphas = alphas / alphas.sum()

    R_iter = 1 if args.one_step else max(1, args.R)

    # initial point v = 0
    v = torch.zeros(D, dtype=torch.float64, device=device)

    #################################
    #         Weiszfeld iterations
    #################################
    for _ in range(R_iter):
        diffs = torch.stack([v - fg for fg in flat_clients_tensor], dim=0)  # (m, D)
        norms = torch.linalg.norm(diffs, dim=1)                    # (m,)
        denom = torch.clamp(norms, min=nu)
        betas = alphas / denom                                     # (m,)
        numerator = sum(betas[i] * flat_clients_tensor[i] for i in range(num_clients))
        v = numerator / betas.sum()

    # Step 2: unflatten to original structure
    aggregated = unflatten(v, client_grads[0])
    return aggregated

def clusters_dissimilarity(clusters, centers):
    n0 = len(clusters[0])
    n1 = len(clusters[1])
    m = n0 + n1 
    cs0 = smp.cosine_similarity(clusters[0])
    cs1 = smp.cosine_similarity(clusters[1])
    mincs0 = np.min(cs0, axis=1)
    mincs1 = np.min(cs1, axis=1)
    ds0 = n0/m * (1 - np.mean(mincs0))
    ds1 = n1/m * (1 - np.mean(mincs1))
    return ds0, ds1

# Get average weights
def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] *(1/sum(marks))
    return w_avg

def average_updates(local_updates, marks):
    num_participating_clients = np.sum(marks)
    total_update = local_updates[0] * marks[0] if marks[0] > 0 else local_updates[1] * marks[1]
        
    # 5A. 遍历剩余客户端，进行累加
    for i in range(1, len(local_updates)):
        if marks[i] > 0:
            total_update += local_updates[i]
        
    # 6A. 计算平均值
    update_avg = total_update / num_participating_clients
    return update_avg

def aggregate_esfl(local_models, global_model, ptypes):
    """
    ESFL: Accelerating Poisonous Model Detection in  Privacy-Preserving Federated Learning
    PPRA: Privacy-Preserving Robust Aggregation
    """
    local_weights = [copy.deepcopy(model).state_dict() for model in local_models]
    m = len(local_models)
    for i in range(m):
        local_models[i] = list(local_models[i].parameters())
    global_model = list(global_model.parameters())
    dw = [None for i in range(m)]
    db = [None for i in range(m)]
    for i in range(m):
        dw[i]= global_model[-2].cpu().data.numpy() - \
            local_models[i][-2].cpu().data.numpy() 

        db[i]= global_model[-1].cpu().data.numpy() - \
            local_models[i][-1].cpu().data.numpy()
        
    dw = np.asarray(dw)
    db = np.asarray(db)
    print(dw.shape)

    "If one class or two classes classification model"
    if len(db[0]) <= 2:
        data = []
        for i in range(m):
            data.append(dw[i].reshape(-1))
    
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        labels = kmeans.labels_

        clusters = {0:[], 1:[]}
        for i, l in enumerate(labels):
            clusters[l].append(data[i])

        good_cl = 0
        cs0, cs1 = clusters_dissimilarity(clusters, kmeans.cluster_centers_)
        if cs0 < cs1:
            good_cl = 1

        scores = np.ones([m])
        for i, l in enumerate(labels):
            # print(ptypes[i], 'Cluster:', l)
            if l != good_cl:
                scores[i] = 0
            
        global_weights = average_weights(local_weights, scores)
        return global_weights

    "For multiclassification models"
    norms = np.linalg.norm(dw, axis = -1) 
    memory = np.sum(norms, axis = 0)
    memory +=np.sum(abs(db), axis = 0)
    max_two_freq_classes = memory.argsort()[-2:]
    data = []
    for i in range(m):
        data.append(dw[i][max_two_freq_classes].reshape(-1))

    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    labels = kmeans.labels_

    clusters = {0:[], 1:[]}
    for i, l in enumerate(labels):
        clusters[l].append(data[i])

    good_cl = 0
    cs0, cs1 = clusters_dissimilarity(clusters, kmeans.cluster_centers_)
    if cs0 < cs1:
        good_cl = 1

    scores = np.ones([m])
    for i, l in enumerate(labels):
        if l != good_cl:
            scores[i] = 0
        
    global_weights = average_weights(local_weights, scores)
    return global_weights

def aggregate_esfl2(client_updates, global_model, args):
    """
    ESFL: Accelerating Poisonous Model Detection in  Privacy-Preserving Federated Learning
    PPRA: Privacy-Preserving Robust Aggregation
    """ 
    m = len(client_updates)
    scores = np.ones([m])
    client_updates_numpy_list = []
    for i, update in enumerate(client_updates):
        update_np = update.cpu().numpy()
        if np.isnan(update_np).any():
            print(f"Client {i} has NaN in updates!")
            scores[i] = 0
        else:
            client_updates_numpy_list.append(update_np)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(client_updates_numpy_list)
    labels = kmeans.labels_

    clusters = {0:[], 1:[]}
    for i, l in enumerate(labels):
        clusters[l].append(client_updates_numpy_list[i])

    print(f"簇 0 的样本数量：{len(clusters[0])}")
    print(f"簇 1 的样本数量：{len(clusters[1])}")
    good_cl = 0
    cs0, cs1 = clusters_dissimilarity(clusters, kmeans.cluster_centers_)
    if cs0 < cs1:
        good_cl = 1

    print(f"最佳簇: {good_cl}")

    for i, l in enumerate(labels):
        if l != good_cl:
            scores[i] = 0
        
    aggr_update = average_updates(client_updates, scores)
    
    return aggr_update

def get_largest_cluster_label(clustering):
    """
    从 DBSCAN 聚类结果中获取拥有最多样本的簇的编号。
    忽略标签为 -1 的噪声点。
    """
    if not hasattr(clustering, 'labels_'):
        return None, 0

    labels = clustering.labels_
    
    # 统计所有标签的频率
    counts = Counter(labels)
    
    # 排除 DBSCAN 的噪声点 (-1)
    if -1 in counts:
        del counts[-1]
        
    if not counts:
        # 如果只有噪声点或没有点
        return None, 0
    
    # 找到计数最大的簇
    # most_common(1) 返回 [ (标签, 数量) ]
    largest_cluster_item = counts.most_common(1)[0]
    
    largest_cluster_label = largest_cluster_item[0]
    max_count = largest_cluster_item[1]
    
    return largest_cluster_label, max_count

def aggregate_my_algo2(client_updates, server_update, args, **kwargs):
    """
    DBSCAN-based Trust Aggregation (DTA)
    注意：在模拟中，我们直接使用明文梯度计算距离和求和。
    在实际部署中，'clustering'步骤使用的是掩码梯度的距离 (Masked Dist)，
    'aggregation'步骤使用的是同态去掩码后的总和 (Unmasked Sum)。
    数学上，Given key diff, Masked Dist == Real Dist。
    """
    a = 1
    b = 1/20
    #c = 1/2
    c = 1/2

    # start_idx, flat_size = get_last_layer_info(global_model)
    client_updates_numpy_list = [update.cpu().numpy() for update in client_updates]
    server_update_numpy = server_update.cpu().numpy()

    # server_target_update = server_update_numpy[start_idx : start_idx + flat_size]

    # 1. Dynamic Eps (base on dist median)
    dists = cdist(client_updates_numpy_list, client_updates_numpy_list, metric='cityblock')
    median_dist = np.median(dists)
    # maybe we can adjust the epsilon with Non-IID args.lamda
    if args.dataset == "mnist" : dynamic_eps = median_dist * args.lamda
    elif args.dataset == "cifar" : dynamic_eps = median_dist * args.lamda
    if np.isnan(dynamic_eps) : dynamic_eps = 1.0

    print(f"eps:{dynamic_eps}, median dist:{median_dist}")
            
    # 2. HDBSCAN cluster
    #    We can get the manhattan distance of masked grads with the key differences
    #    dist_ij = ||masked_grad_i - masked_grad_j - F(key_i - key_j, b)||
    #X = flat_clients.detach().numpy()
    #clustering = hdbscan.HDBSCAN(min_cluster_size=2, metric='manhattan').fit(X)
    clustering = DBSCAN(eps=dynamic_eps, min_samples=2, metric='manhattan').fit(client_updates_numpy_list)
    labels = clustering.labels_

    n_clusters_ = len(set(labels))
    n_noise_ = list(labels).count(-1)

    print(f'DSCAN计的聚类数量: {n_clusters_}')
    print(f'DSCAN估计的噪声点数量: {n_noise_}')

    unique_labels = set(labels) - {-1}

    # if not unique_labels:
    #     return server_update

    if not unique_labels:
        return server_update

    cluster_means = []
    raw_scores = []
    
    # 3. trust evaluation
    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]
        cluster_vectors = []
        # use the mean gradients to represent the cluster
        # we can get the sum of masked grads with the keys' sum as:
        # mean of cluster grads = 1/n * (sum_of_masked_grads - F(sum_of_keys, b))
        for i in indices:
            cluster_vectors.append(client_updates_numpy_list[i])  # 前半部分
        mean_vec = np.mean(cluster_vectors, axis=0)

        # cosine similarity with root gradient
        # 1. 方向得分，Sigmoid (a 越大，对方向越敏感)
        sim = 1 - cosine(mean_vec, server_update_numpy)
        s_sim = 1 / (1 + np.exp(-a * sim))

        # 2. 范数得分 (无阈值)
        cluster_norm = np.linalg.norm(mean_vec, axis = -1)
        server_grad_norm = np.linalg.norm(server_update_numpy, axis = -1)
        norm_ratio = cluster_norm / (server_grad_norm + 1e-9)
        s_norm = np.exp(-b * (np.abs(norm_ratio - 1)))

        # 规模得分
        s_size = len(indices) / args.num_clients
        s_size = s_size ** c

        score = s_sim * s_norm * s_size  # weight by cluster size

        cluster_means.append(mean_vec)
        raw_scores.append(score)

        print(f"label: {label} confidence: {score}")
        print(f"cluster mean vector norm: {cluster_norm}")
        print(f"server update norm: {server_grad_norm}")
        print(f"s_sim: {s_sim} s_norm: {s_norm}")
        print(f"client indices: {indices}")
    
    # 4. weighted aggregation
    raw_scores = np.array(raw_scores) # 放大分数以增强区分度
    # 这里的 T 是温度参数。T 越小，权重越向高分簇集中；T 越大，权重越平均。
    # 默认 T=1.0。如果你想让模型更果断地剔除差簇，可以把 raw_scores 乘以一个放大系数。
    T = 0.05
    exp_scores = np.exp(raw_scores / T)
    softmax_confs = exp_scores / np.sum(exp_scores)

    final_update = np.zeros_like(server_update_numpy)

    for i, mean_vec in enumerate(cluster_means):
        final_update += softmax_confs[i] * mean_vec
        print(f"Cluster {i} Softmax Weight: {softmax_confs[i]:.4f}")
    
    aggr_update_tensor = torch.tensor(final_update).to(torch.float32)

    return aggr_update_tensor

def k_medoids(X, n_clusters, max_iter=100, random_state=None):
    rng = np.random.RandomState(random_state)
    m = X.shape[0]
    if m < n_clusters:
        return np.arange(m)
    
    medoid_indices = rng.choice(m, n_clusters, replace=False)
    medoids = X[medoid_indices]
    labels = np.zeros(m)
    
    for _ in range(max_iter):
        dists = cdist(X, medoids, metric='euclidean')
        labels = np.argmin(dists, axis=1)
        
        new_medoid_indices = []
        for k in range(n_clusters):
            cluster_points_idx = np.where(labels == k)[0]
            if len(cluster_points_idx) == 0:
                new_medoid_indices.append(medoid_indices[k])
                continue
            
            cluster_X = X[cluster_points_idx]
            intra_dists = cdist(cluster_X, cluster_X, metric='euclidean')
            sums = np.sum(intra_dists, axis=1)
            best_idx_in_cluster = np.argmin(sums)
            new_medoid_indices.append(cluster_points_idx[best_idx_in_cluster])
        
        new_medoid_indices = np.array(new_medoid_indices)
        if np.all(new_medoid_indices == medoid_indices):
            break
        medoid_indices = new_medoid_indices
        medoids = X[medoid_indices]
        
    return labels

def aggregate_my_algo(client_updates, server_update, args, **kwargs):
    """
    DBHC-based Aggregation (aggregator_my_algo)
    Replaces DBSCAN with DBHC for clustering.
    Reference: aggregate_my_algo2
    """
    client_updates_numpy_list = []
    
    for i, update in enumerate(client_updates):
        update_np = update.cpu().numpy()
        if np.isnan(update_np).any():
             print(f"Client {i} has NaN in updates! Skipping.")
        else:
            client_updates_numpy_list.append(update_np)
            
    if not client_updates_numpy_list:
        return server_update

    server_update_numpy = server_update.cpu().numpy()
    
    # Use n_clusters=None to trigger Auto-k logic and visualization
    dbhc = DBHC(n_clusters=None, plot_merge_cost=True)
    dbhc.fit(client_updates_numpy_list)
    labels = dbhc.labels_


    # # --- Clustering Structure Visualization (Sorted Distance Matrix) ---
    # try:
    #     dists = cdist(client_updates_numpy_list, client_updates_numpy_list, metric='cityblock')
    #     median_dist = np.median(dists)
    #     lamda = getattr(args, 'lamda', 1.0)
    #     dynamic_eps = median_dist * lamda
    #     if np.isnan(dynamic_eps):
    #         dynamic_eps = 1.0
        
    #     dbscan_labels = DBSCAN(eps=dynamic_eps, min_samples=2, metric='manhattan').fit_predict(client_updates_numpy_list)
        
    #     kmeans = KMeans(n_clusters=2, random_state=0).fit(client_updates_numpy_list)
    #     kmeans_labels = kmeans.labels_

    #     data_matrix = np.array(client_updates_numpy_list)
    #     dist_matrix = cdist(data_matrix, data_matrix, metric='euclidean')
        
    #     fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
    #     algorithms = [
    #         ("DBHC", labels),
    #         ("DBSCAN", dbscan_labels),
    #         ("K-Means", kmeans_labels)
    #     ]
        
    #     for ax, (algo_name, algo_labels) in zip(axes, algorithms):
    #         sorted_idx = np.argsort(algo_labels)
            
    #         sorted_dist = dist_matrix[sorted_idx][:, sorted_idx]
            
    #         unique, counts = np.unique(algo_labels, return_counts=True)
    #         stats_parts = []
    #         for u, c in zip(unique, counts):
    #             label_str = "Noise" if u == -1 else f"Clust {u}"
    #             stats_parts.append(f"{label_str}:{c}")
    #         stats_str = ", ".join(stats_parts)
            
    #         sns.heatmap(sorted_dist, ax=ax, cmap='viridis', square=True, 
    #                     cbar=True, xticklabels=False, yticklabels=False)
            
    #         ax.set_title(f"{algo_name} Structure\n({stats_str})")
    #         ax.set_ylabel("Clients (Sorted)")
    #         ax.set_xlabel("Clients (Sorted)")
            
    #     plt.tight_layout()
    #     plt.savefig("clustering_structure_heatmap.png")
    #     plt.close()
    #     print("Saved clustering structure heatmap to clustering_structure_heatmap.png")
        
    # except Exception as e:
    #     print(f"Error during clustering comparison visualization: {e}")
    # # --- Clustering Comparison Block End ---
    
    unique_labels = set(labels) - {-1}
    
    if not unique_labels:
        print("DBHC found no clusters (all noise). Returning server update.")
        return server_update
        
    cluster_means = []
    raw_scores = []
    
    # Trust evaluation
    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]
        
        # Get updates belonging to this cluster
        cluster_vectors = [client_updates_numpy_list[i] for i in indices]
        mean_vec = np.mean(cluster_vectors, axis=0)
        
        # Calculate score (same as aggregate_my_algo2)
        # 1. Cosine Similarity
        # Note: cosine returns distance (1 - sim), so 1 - cosine is similarity
        sim = 1 - cosine(mean_vec, server_update_numpy)
        s_sim = 1 / (1 + np.exp(-args.agg_a * sim))
        
        # 2. Norm Score
        cluster_norm = np.linalg.norm(mean_vec, axis=-1)
        server_grad_norm = np.linalg.norm(server_update_numpy, axis=-1)
        norm_ratio = cluster_norm / (server_grad_norm + 1e-9)
        s_norm = np.exp(-args.agg_b * (np.abs(norm_ratio - 1)))
        
        # 3. Size Score
        s_size = len(indices) / args.num_clients 
        s_size = s_size ** args.agg_c
        
        score = s_sim * s_norm * s_size
        
        cluster_means.append(mean_vec)
        raw_scores.append(score)
        
        print(f"DBHC Label: {label}, Score: {score:.4f}, Size: {len(indices)}")
        print(f"cluster mean vector norm: {cluster_norm}")
        print(f"server update norm: {server_grad_norm}")
        print(f"s_sim: {s_sim} s_norm: {s_norm}")
        print(f"client indices: {indices}")
    # Weighted Aggregation
    if not raw_scores:
         return server_update

    raw_scores = np.array(raw_scores)
    T = args.agg_T
    exp_scores = np.exp(raw_scores / T)
    softmax_confs = exp_scores / np.sum(exp_scores)
    
    final_update = np.zeros_like(server_update_numpy)
    
    for i, mean_vec in enumerate(cluster_means):
        weight = softmax_confs[i]
        final_update += weight * mean_vec
        print(f"Cluster {i} Weight: {weight:.4f}")
        
    aggr_update_tensor = torch.tensor(final_update).to(torch.float32)
    return aggr_update_tensor
