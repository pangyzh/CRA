import torch
import copy
import time
import pandas as pd
from torch import nn
import torchvision as tv
import numpy as np
import scipy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import os
from datetime import datetime
from args import args_parser
from dataloader import get_dataset, CNN_MNIST, CNN_CIFAR
from clients import LocalUpdate
from attacker import configure_malicious_clients, perturb_collusion,poison_minmax
from aggregator import aggregate_fedavg, aggregate_krum, aggregate_fltrust, aggregate_my_algo, aggregate_my_algo2,aggregate_esfl,aggregate_esfl2, aggregate_priroagg_rfa

def evaluate(net, dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    args = args_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {args.device}, Dataset: {args.name_dataset}, Algo: {args.agg_method}, Attack: {args.attack_type}")

    train_dataset, test_dataset, user_groups, root_data, num_channels = get_dataset(args)

    num_classes = 10
    if args.name_dataset == 'mnist':
        global_model = CNN_MNIST().to(args.device)
    elif args.name_dataset == 'cifar':
        global_model = CNN_CIFAR().to(args.device)
    # global_model.train()

    malicious_users = set()
    if args.attack_type != "none":
        malicious_users = configure_malicious_clients(args)
        print(f"Malicious Clients ({len(malicious_users)}): {list(malicious_users)[:10]}...")

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    
    metrics = {
        'malicious_clients': [],
        'round': [],
        'accuracy': [],
        'time': []
    }

    for round_idx in range(args.rounds):
        start_time = time.time()
        local_weights = []
        local_grads = []
        local_models = []
        local_updates = []
        
        # for ESFL
        # global_model_state = copy.deepcopy(global_model.state_dict())
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        # --- Client Training ---
        idxs_users = range(args.num_clients)
        
        for idx in idxs_users:
            is_mal = idx in malicious_users
            local_client = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], 
                                     device=args.device, is_malicious=is_mal)
            

            grads, model, update = local_client.train(copy.deepcopy(global_model).to(args.device))
            local_grads.append(grads)
            local_models.append(model)
            local_updates.append(update)
        
        # --- Server Training (Root Data) ---
        root_grads = None
        if args.agg_method in ["fltrust", "my_algo"]:
            root_client = LocalUpdate(args=args, dataset=root_data, idxs=list(range(len(root_data))), 
                                    device=args.device, is_malicious=False)
            root_grads, root_model, root_update = root_client.train(copy.deepcopy(global_model).to(args.device))
        
        if args.attack_type == "alie":
            # ----------------------------
            # A little is enough Attack（ALIE）
            # ----------------------------
            perturb_collusion(local_grads, malicious_users, args)

        if args.attack_type == "min-max":
            perturb_collusion(local_grads, malicious_users, args)

        # --- Aggregation ---
        if args.agg_method == "fedavg":
            agg_grads = aggregate_fedavg(local_grads)
        elif args.agg_method == "krum":
            agg_grads = aggregate_krum(local_grads, args)
        elif args.agg_method == "fltrust":
            agg_grads = aggregate_fltrust(local_grads, root_grads)
        elif args.agg_method == "rfa":
            agg_grads = aggregate_priroagg_rfa(local_grads, args)
        elif args.agg_method == "esfl":
            # ESFL needs global_model_state to calculate the confidence
            #global_weights = aggregate_esfl(copy.deepcopy(local_models), copy.deepcopy(global_model), args)
            agg_update = aggregate_esfl2(local_updates, copy.deepcopy(global_model).to(args.device), args)
        elif args.agg_method == "my_algo":
            agg_update = aggregate_my_algo2(local_updates, root_update, copy.deepcopy(global_model).to(args.device), args)
        else:
            raise ValueError("Unknown aggregation method")
            
        # --- Global Update ---
        #if args.agg_method == "esfl":
        #
        if args.agg_method in ["esfl", "my_algo"]:
            with torch.no_grad():
                # for param, update_val in zip(global_model.parameters(), agg_update):
                    # 直接相加。
                    # update_val = W_final - W_init，所以 W_new = W_init + update_val
                # param.data.add_(agg_update)
                vector_to_parameters(agg_update + initial_global_model_params, global_model.parameters())
        else:
            with torch.no_grad():
                for param, grad_val in zip(global_model.parameters(), agg_grads):
                    # 1. 确保设备一致
                    grad_val = grad_val.to(param.device)
                    
                    # 2. 执行减法 (Gradient Descent)
                    # W_new = W_old - lr * gradient
                    param.data.sub_(grad_val * args.lr)
                
        end_time = time.time()
        round_time = end_time - start_time
        
        # --- Evaluation ---
        global_model.float()
        test_acc = evaluate(global_model, test_dataset, args)
        print(f"Round {round_idx+1}/{args.rounds} | Accuracy: {test_acc:.2f}% | Time: {round_time:.2f}s")
        
        metrics['malicious_clients'].append(malicious_users)
        metrics['round'].append(round_idx+1)
        metrics['accuracy'].append(test_acc)
        metrics['time'].append(round_time)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dist_str = "iid" if args.iid else f"noniid_alpha{args.alpha}"
    file_name = f"{args.name_dataset}_{args.agg_method}_{args.attack_type}_mal{args.mal_prop}_{dist_str}_{timestamp}.csv"
    
    save_path = os.path.join(args.out_path, file_name)
    df = pd.DataFrame(metrics)
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()