import torch
import numpy as np
import copy

def configure_malicious_clients(args):
    """随机选择攻击者索引"""
    num_mal = int(args.num_clients * args.mal_prop)
    mal_indices = np.random.choice(range(args.num_clients), num_mal, replace=False)
    return set(mal_indices)

def perturb_data(data, target, args):
    """数据层面的攻击：如 Label Flipping"""
    if args.attack_type == "label_flip":
        # 简单策略：y = 9 - y
        target = 9 - target
    return data, target

def perturb_gradients(grads, args):
    """梯度层面的攻击"""

    # ----------------------------
    # 1. Gaussian Noise
    # ----------------------------
    if args.attack_type == "gaussian_noise":
        # 添加高斯噪声，使梯度变得无意义
        new_grads = []
        for g in grads:
            noise = torch.randn_like(g) * 5.0 # 强度可调
            new_grads.append(g + noise)
        return new_grads
        
    # ----------------------------
    # 2. Sign-Flip
    # ----------------------------
    elif args.attack_type == "sign_flip":
        # 梯度符号反转
        new_grads = [-g * 2.0 for g in grads] # 放大并反转
        return new_grads

    # ----------------------------
    # 3. A little is enough Attack（ALIE）
    # ----------------------------
    elif args.attack_type == "alie":
        new_grads = []
        epsilon = getattr(args, "alie_eps", 0.01)   # 默认扰动系数，可调
        scale = getattr(args, "alie_scale", 0.1)    # 默认梯度缩放

        for g in grads:
            # 将梯度缩小成很小，让贡献几乎消失
            small_grad = g * scale

            # 再添加微弱、方向性的噪声
            noise = epsilon * torch.sign(g)  # 方向依然沿梯度方向
            new_grads.append(small_grad + noise)

        return new_grads

    # ----------------------------
    # 4. Free-rider
    # ----------------------------
    elif args.attack_type == "free_rider":
        new_grads = []
        noise_scale = getattr(args, "fr_noise", 1e-5)  # 几乎不训练
        for g in grads:
            fake_grad = torch.randn_like(g) * noise_scale
            new_grads.append(fake_grad)
        return new_grads

    return grads