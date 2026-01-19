import torch
import numpy as np
import copy
import scipy
from utils import flatten, unflatten, get_mean_and_std
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def configure_malicious_clients(args):
    """随机选择攻击者索引"""
    num_mal = int(args.num_clients * args.mal_prop)
    mal_indices = np.random.choice(range(args.num_clients), num_mal, replace=False)
    return set(mal_indices)

def solve_cosine_coordinate(m, cl, target_cs_val, k):
    # 1. 强制约束目标值在合理区间 
    target_cs_val = np.clip(target_cs_val, -0.99, 0.99)
    
    cl_norm = np.linalg.norm(cl)
    C = target_cs_val * cl_norm
    
    dot_other = np.dot(m, cl) - m[k] * cl[k]
    norm_sq_other = np.sum(m**2) - m[k]**2
    
    A = cl[k]**2 - C**2
    B = 2 * dot_other * cl[k]
    D = dot_other**2 - C**2 * norm_sq_other
    
    # 2. 检查判别式 
    discriminant = B**2 - 4 * A * D
    if discriminant < 0:
        # 如果无解，尝试降低目标值再次求解，或直接跳过该维度
        return None 
    
    # 3. 处理一次方程情况
    if abs(A) < 1e-9:
        return [-D / B] if abs(B) > 1e-9 else None
        
    roots = [(-B + np.sqrt(discriminant)) / (2 * A), (-B - np.sqrt(discriminant)) / (2 * A)]
    return roots

def perturb_data(data, target, args):
    """数据层面的攻击：如 Label Flipping"""
    if args.attack_type == "label_flip":
        # 简单策略：y = 9 - y
        target = 9 - target
    return data, target

def perturb_gradients(grads, update, args):
    """梯度层面的攻击"""

    # ----------------------------
    # Gaussian Noise
    # ----------------------------
    if args.attack_type == "gaussian_noise":
        # 添加高斯噪声，使梯度变得无意义
        # noise = torch.normal(mean=0.5, std=1, size=grads.shape).to(grads.device)
        # new_grads = grads + noise #scaling
        # update = -new_grads * args.lr
        noise = torch.normal(mean=0.0, std=1, size=update.shape).to(update.device)
        update += noise
        new_grads = -update / args.lr
        return new_grads, update


def perturb_collusion(grads_list, update_list, malicious_users, root_grads, args):

    # grads_flatten = [flatten(g) for g in grads_list]
    # all_grads_flatten = torch.stack(grads_flatten)
    # gradient_mean = torch.mean(all_grads_flatten, dim=0)
    # gradient_std = torch.std(all_grads_flatten, dim=0)
    gradient_mean, gradient_std = get_mean_and_std(grads_list)

    if args.attack_type == "alie":
        # ----------------------------
        # A little is enough Attack（ALIE）
        # ----------------------------
        # parameters for ALIE attack
        # new_grads = []
        # users_grad = []
        # n = args.num_clients
        # m = int(args.num_clients * args.mal_prop)
        # if m > n/2:
        #     m = np.floor(n/2)
        # s = np.floor(n / 2 + 1) - m
        # cdf_value = (n - m - s) / (n - m)
        # z_max = scipy.stats.norm.ppf(cdf_value)
        # if z_max < args.alie_zmax:
        #     z_max = args.alie_zmax
        # print(f"zmax:{z_max}")

        for idx in range(len(grads_list[0])):
            bad_gradient = gradient_mean[idx] - gradient_std[idx] * args.alie_zmax
            bad_update = -args.lr * bad_gradient

            for user_idx in malicious_users:
                grads_list[user_idx][idx].copy_(bad_gradient)
                update_list[user_idx][idx].copy_(bad_update)
        # bad_gradient_flat = gradient_mean - gradient_std * args.alie_zmax
        # bad_update_flat = -args.lr * bad_gradient_flat
        # bad_gradient_layers = unflatten(bad_gradient_flat, grads_list[0])
        # bad_update_layers = unflatten(bad_update_flat, update_list[0])

        # # 批量更新恶意用户的梯度和更新量
        # for user_idx in malicious_users:
        #     for layer_idx in range(len(grads_list[user_idx])):
        #         # 使用原地操作 copy_ 确保内存更新
        #         grads_list[user_idx][layer_idx].copy_(bad_gradient_layers[layer_idx])
        #         update_list[user_idx][layer_idx].copy_(bad_update_layers[layer_idx])

    elif args.attack_type == "min-max":
        flat_clients = torch.stack([flatten(g) for g in grads_list])

        dists = torch.cdist(flat_clients, flat_clients, p=1)
        max_dist = torch.max(dists)

        benign_vector = flatten(gradient_mean)
        #benign_vector = gradient_mean

        # perturbation vector: Inverse unit vector
        deviation = - benign_vector/torch.norm(benign_vector)

        # if dev_type == 'unit_vec':
        #     deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        # elif dev_type == 'sign':
        #     deviation = torch.sign(model_re)
        # elif dev_type == 'std':
        #     deviation = torch.std(all_updates, 0)

        lamda = 40
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        while abs(lamda_succ - lamda) > threshold_diff:
            mal_grad = (benign_vector + lamda * deviation)
            distance = torch.norm((flat_clients - mal_grad), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_dist:
                print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2
                print('continue')

            lamda_fail = lamda_fail / 2

        unflatten(deviation, grads_list[0])

        #noise = torch.normal(mean=0.0, std=1, size=m.shape)

        # bad_gradient_flat = gradient_mean + args.min_max_scale * lamda_succ * deviation
        # bad_update_flat = -args.lr * bad_gradient_flat
        # bad_gradient_layers = unflatten(bad_gradient_flat, grads_list[0])
        # bad_update_layers = unflatten(bad_update_flat, update_list[0])

        # 批量更新恶意用户的梯度和更新量
        # for user_idx in malicious_users:
        #     for layer_idx in range(len(grads_list[user_idx])):
        #         # 使用原地操作 copy_ 确保内存更新
        #         grads_list[user_idx][layer_idx].copy_(bad_gradient_layers[layer_idx])
        #         update_list[user_idx][layer_idx].copy_(bad_update_layers[layer_idx])
        for idx in range(len(grads_list[0])):
            bad_gradient = gradient_mean[idx] + args.min_max_scale * lamda_succ * deviation[idx]
            bad_update = -args.lr * bad_gradient

            for user_idx in malicious_users:
                grads_list[user_idx][idx].copy_(bad_gradient)
                update_list[user_idx][idx].copy_(bad_update)

    elif args.attack_type == "sine":
        """
        Sine 攻击实现 [cite: 241, 353]
        :param v_b: 平均良性模型更新 (Average Benign Update) 
        :param v_cl: 受控领导者更新 (Compromised Leader Update) 
        :param gamma_c: 余弦缩放因子 [cite: 371, 422]
        :param gamma_n: 范数缩放因子 [cite: 371, 422]
        """
        gamma_c = args.gamma_c
        gamma_n = args.gamma_n
        v_b = flatten(gradient_mean)
        v_cl = flatten(root_grads)
        m = np.copy(v_b) # 初始化恶意更新为良性更新 [cite: 253]
        d = len(m)
        cl_norm = np.linalg.norm(v_cl)
    
        # 1. 寻找感兴趣的坐标 (按差异绝对值降序排列) [cite: 259-261, 354]
        diff = np.abs(v_b - v_cl)
        imp_k = np.argsort(-diff)
    
        # 计算初始余弦相似度并设置目标
        initial_cs = np.dot(m, v_cl) / (np.linalg.norm(m) * cl_norm)
        target_cs = gamma_c * initial_cs
    
        current_cs_mb = 1.0 # m 与 b 的相似度，初始化为 1 [cite: 256]
    
        # 2. 逐坐标迭代修改 [cite: 262, 356]
        for k in imp_k[:int(0.1 * d)]: # 论文提到 k << d，此处取前 10% 维度 [cite: 369]
            # 记录旧值以便回滚
            old_val = m[k]
            
            # 求解满足余弦相似度的新坐标 [cite: 250, 358]
            potential_roots = solve_cosine_coordinate(m, v_cl, target_cs, k)
            
            if isinstance(potential_roots, list):
                # 论文逻辑：寻找能让 cos(m, v_b) 减小的根 [cite: 268, 330]
                for root in potential_roots:
                    m[k] = root
                    new_cs_mb = np.dot(m, v_b) / (np.linalg.norm(m) * np.linalg.norm(v_b))
                    new_norm_m = np.linalg.norm(m)
                    
                    # 约束检查：
                    # 1. 对良性更新的相似度必须下降 [cite: 268, 330]
                    # 2. 范数必须在可信范围内 [cite: 270, 360]
                    norm_min = cl_norm / gamma_n
                    norm_max = cl_norm * gamma_n
                    
                    if new_cs_mb < current_cs_mb and norm_min < new_norm_m < norm_max:
                        current_cs_mb = new_cs_mb
                        break # 接受该修改
                    else:
                        m[k] = old_val # 回滚
        
        # unflatten(m, grads_list[0])
        noise = torch.normal(mean=0.0, std=1, size=m.shape)
        for user_idx in malicious_users:
            grads_list[user_idx] = m + noise.numpy()
            update_list[user_idx] = -args.lr * grads_list[user_idx]
        # for idx in range(len(grads_list[0])):
        #     #noise = torch.normal(mean=0.0, std=1, size=m[idx].shape)
        #     bad_gradient = m[idx]
        #     bad_update = -args.lr * bad_gradient

        #     for user_idx in malicious_users:
        #         grads_list[user_idx][idx].copy_(bad_gradient)
        #         update_list[user_idx][idx].copy_(bad_update)

    return None
