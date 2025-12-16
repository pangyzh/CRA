import torch
import numpy as np
import copy
import scipy
from utils import flatten, unflatten

def configure_malicious_clients(args):
    """随机选择攻击者索引"""
    num_mal = int(args.num_clients * args.mal_prop)
    mal_indices = np.random.choice(range(args.num_clients), num_mal, replace=False)
    return set(mal_indices)

def get_mean_and_std(gradient_list):

    gradient_mean, gradient_std = [], []
    for idx in range(len(gradient_list[0])):
        layer_gradient_stack = torch.stack([gradient_list[i][idx] for i in range(len(gradient_list))], dim=0)
        gradient_mean.append(torch.mean(layer_gradient_stack, dim=0))
        gradient_std.append(torch.std(layer_gradient_stack, dim=0))
    
    return gradient_mean, gradient_std

def perturb_data(data, target, args):
    """数据层面的攻击：如 Label Flipping"""
    if args.attack_type == "label_flip":
        # 简单策略：y = 9 - y
        target = 9 - target
    return data, target

def perturb_gradients(grads, args):
    """梯度层面的攻击"""

    # ----------------------------
    # Gaussian Noise
    # ----------------------------
    if args.attack_type == "gaussian_noise":
        # 添加高斯噪声，使梯度变得无意义
        new_grads = []
        for g in grads:
            noise = torch.normal(mean=0.0, std=1, size=g.shape).to(g.device)
            out = g + noise #scaling
            new_grads.append(out)
        return new_grads
        
    # ----------------------------
    # Sign-Flip
    # ----------------------------
    elif args.attack_type == "sign_flip":
        # 梯度符号反转
        new_grads = [-g * 2.0 for g in grads] # 放大并反转
        return new_grads

    # ----------------------------
    # Free-rider
    # ----------------------------
    elif args.attack_type == "free_rider":
        new_grads = []
        noise_scale = getattr(args, "fr_noise", 1e-5)  # 几乎不训练
        for g in grads:
            fake_grad = torch.randn_like(g) * noise_scale
            new_grads.append(fake_grad)
        return new_grads

    return grads

def perturb_collusion(grads_list, malicious_users, args):

    gradient_mean, gradient_std = get_mean_and_std(grads_list)

    if args.attack_type == "alie":
        # ----------------------------
        # A little is enough Attack（ALIE）
        # ----------------------------
        # parameters for ALIE attack
        new_grads = []
        users_grad = []
        n = args.num_clients
        m = int(args.num_clients * args.mal_prop)
        s = np.floor(n / 2 + 1) - m
        cdf_value = (n - m - s) / (n - m)
        z_max = scipy.stats.norm.ppf(cdf_value)

        print(f"zmax:{z_max}")

        for idx in range(len(grads_list[0])):
            bad_gradient = gradient_mean[idx] - gradient_std[idx] * z_max

            for user_idx in malicious_users:
                grads_list[user_idx][idx].copy_(bad_gradient)
    
    elif args.attack_type == "min-max":
        flat_clients = torch.stack([flatten(g) for g in grads_list])

        dists = torch.cdist(flat_clients, flat_clients, p=1)
        max_dist = torch.max(dists)

        benign_vector = flatten(gradient_mean)

        # perturbation vector: Inverse unit vector
        deviation = - benign_vector/torch.norm(benign_vector)

        # if dev_type == 'unit_vec':
        #     deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        # elif dev_type == 'sign':
        #     deviation = torch.sign(model_re)
        # elif dev_type == 'std':
        #     deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([20.0]).float().cpu()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_grad = (benign_vector + lamda * deviation)
            distance = torch.norm((flat_clients - mal_grad), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_dist:
                print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        unflatten(deviation, flat_clients[0])

        for idx in range(len(grads_list[0])):
            bad_gradient = gradient_mean[idx] + lamda_succ * deviation[idx]

            for user_idx in malicious_users:
                grads_list[user_idx][idx].copy_(bad_gradient)

    return None

def poison_minmax(grads_list, malicious_users, args):
    """
        min max attack, num_corrupt should be 1 in this case, just copy malicious behavior if num_corrupt is greater than one
        v_avg + \gamma*v_p -v_i
    """

    flat_clients = torch.stack([flatten(g) for g in grads_list])

    dists = torch.cdist(flat_clients, flat_clients, p=1)
    max_dist = torch.max(dists)

    gradient_mean, gradient_std = get_mean_and_std(grads_list)
    benign_vector = flatten(gradient_mean)

    # perturbation vector: Inverse unit vector
    deviation = - benign_vector/torch.norm(benign_vector)

    # if dev_type == 'unit_vec':
    #     deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    # elif dev_type == 'sign':
    #     deviation = torch.sign(model_re)
    # elif dev_type == 'std':
    #     deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([20.0]).float().cpu()
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_grad = (benign_vector + lamda * deviation)
        distance = torch.norm((flat_clients - mal_grad), dim=1) ** 2
        max_d = torch.max(distance)
        
        if max_d <= max_dist:
            print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    #bad_gradient = (benign_vector + lamda_succ * deviation)
    
    unflatten(deviation, flat_clients[0])

    for idx in range(len(grads_list[0])):
        bad_gradient = gradient_mean[idx] + lamda_succ * deviation[idx]

        for user_idx in malicious_users:
            grads_list[user_idx][idx].copy_(bad_gradient)

    return None