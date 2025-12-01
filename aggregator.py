import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def aggregate_fedavg(client_grads, **kwargs):
    """FedAvg"""
    num_clients = len(client_grads)
    weighted_grads = [torch.zeros_like(g) for g in client_grads[0]]
    for grads in client_grads:
        for i, g in enumerate(grads):
            weighted_grads[i] += g / num_clients
    return weighted_grads

def aggregate_krum(client_grads, args, **kwargs):
    """Krum"""
    f = int(args.num_clients * args.mal_prop)
    n = len(client_grads)
    if n <= 2 * f + 2: 
        # Fallback
        return aggregate_fedavg(client_grads)
        
    flat_grads = torch.stack([flatten(g) for g in client_grads])
    scores = []
    # calculate the dist matrix
    dists = torch.cdist(flat_grads, flat_grads, p=2)
    
    for i in range(n):
        # get the top k nearest neighbor
        k_nearest = dists[i].topk(n - f - 1, largest=False).values
        scores.append(k_nearest.sum())
        
    best_idx = torch.argmin(torch.tensor(scores))
    return client_grads[best_idx]

def aggregate_fltrust(client_grads, server_grad, **kwargs):
    """FLTrust"""
    flat_server = flatten(server_grad)
    norm_server = torch.norm(flat_server)
    
    weighted_sum = torch.zeros_like(flat_server)
    total_weight = 0.0
    
    for grads in client_grads:
        flat_client = flatten(grads)
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
        
    return unflatten(final_flat, server_grad)

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


def aggregate_esfl_ppra(client_grads, global_model_state, args, **kwargs):
    """
    ESFL: Accelerating Poisonous Model Detection in  Privacy-Preserving Federated Learning
    PPRA: Privacy-Preserving Robust Aggregation
    """

    flat_clients_tensor = torch.stack([flatten(g) for g in client_grads])
    flat_clients_np = flat_clients_tensor.detach().numpy()
    
    num_clients = len(client_grads)
    K = args.num_clusters #default to 2

    kmeans = KMeans(n_clusters=K, random_state=args.seed, n_init='auto')
    kmeans.fit(flat_clients_np)
    labels = kmeans.labels_ 

    global_model_flat_list = [v.flatten() for v in global_model_state.values()]
    global_model_flat_tensor = torch.cat(global_model_flat_list)
    global_model_flat_np = global_model_flat_tensor.detach().numpy().reshape(1, -1)
    
    best_cluster_label = -1
    max_confidence = -float('inf')
    
    # select the best cluster
    for label in range(K):
        cluster_indices = np.where(labels == label)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        cluster_grads_np = flat_clients_np[cluster_indices]
        similarities = cosine_similarity(cluster_grads_np, global_model_flat_np).flatten()
        confidence = np.mean(np.maximum(0, similarities))
        
        if confidence > max_confidence:
            max_confidence = confidence
            best_cluster_label = label

    best_cluster_indices = np.where(labels == best_cluster_label)[0]
    best_cluster_grads_np = flat_clients_np[best_cluster_indices]
    weights = np.maximum(0, cosine_similarity(best_cluster_grads_np, global_model_flat_np).flatten())
    total_weight = np.sum(weights)
    normalized_weights = weights / total_weight
    
    weighted_flat_agg_np = np.sum(best_cluster_grads_np * normalized_weights[:, np.newaxis], axis=0)
    weighted_flat_agg_tensor = torch.from_numpy(weighted_flat_agg_np).to(flat_clients_tensor.device).float()
    
    return unflatten(weighted_flat_agg_tensor, client_grads[0])

def aggregate_my_algo(client_grads, server_grad, args, **kwargs):
    """
    DBSCAN-based Trust Aggregation (DTA)
    注意：在模拟中，我们直接使用明文梯度计算距离和求和。
    在实际部署中，'clustering'步骤使用的是掩码梯度的距离 (Masked Dist)，
    'aggregation'步骤使用的是同态去掩码后的总和 (Unmasked Sum)。
    数学上，Given key diff, Masked Dist == Real Dist。
    """
    flat_server = flatten(server_grad)
    flat_clients = torch.stack([flatten(g) for g in client_grads])
    
    # 1. Dynamic Eps (base on dist median)
    with torch.no_grad():
        dists = torch.cdist(flat_clients, flat_clients, p=1)
        median_dist = torch.median(dists)
        # maybe we can adjust the epsilon with Non-IID args.alpha
        if args.iid : dynamic_eps = median_dist.item() * 1.5
        elif args.name_dataset == "mnist" : dynamic_eps = median_dist.item() * args.alpha
        elif args.name_dataset == "cifar" : dynamic_eps = median_dist.item() * args.alpha
        if dynamic_eps < 1e-3: dynamic_eps = 1.0
            
    # 2. HDBSCAN cluster
    #    We can get the manhattan distance of masked grads with the key differences
    #    dist_ij = ||masked_grad_i - masked_grad_j - F(key_i - key_j, b)||
    X = flat_clients.detach().numpy()
    #clustering = hdbscan.HDBSCAN(min_cluster_size=2, metric='manhattan').fit(X)
    clustering = DBSCAN(eps=dynamic_eps, min_samples=2, metric='manhattan').fit(X)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}

    if not unique_labels:
        return server_grad
    
    cluster_means = []
    confidences = []
    
    # 3. trust evaluation
    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]
        
        # use the mean gradients to represent the cluster
        # we can get the sum of masked grads with the keys' sum as:
        # mean of cluster grads = 1/n * (sum_of_masked_grads - F(sum_of_keys, b))
        cluster_vectors = flat_clients[indices]
        mean_vec = torch.mean(cluster_vectors, dim=0)
        
        # cosine similarity with root gradient
        sim = F.cosine_similarity(mean_vec, flat_server, dim=0)

        # ReLU
        conf = F.relu(sim)

        conf = conf * len(indices)/args.num_clients  # weight by cluster size
        
        cluster_means.append(mean_vec)
        confidences.append(conf)

        print(f"label: {label} confidence: {conf}")
        print(f"client indices: {indices}")
    
    # 4. weighted aggregation
    conf_tensor = torch.stack(confidences)
    total_conf = conf_tensor.sum()
    
    final_flat = torch.zeros_like(flat_server)
    
    if total_conf > 0.01:
        # normaliztion the cluster confidence
        normalized_confs = conf_tensor / total_conf
        for i, mean_vec in enumerate(cluster_means):
            final_flat += normalized_confs[i] * mean_vec
    else:
        # two possible reasons for this case (total_conf <= 0.01):
        # 1. there are no benigh clusters,
        # 2. the model has been already converaged, so that the gradients is too small to
        #    calculate the cosine similarity,
        # so we just return server model here.
        final_flat = flat_server
        
    return unflatten(final_flat, server_grad)