import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from attacker import perturb_data, perturb_gradients

class LocalUpdate:
    def __init__(self, args, dataset, idxs, device, is_malicious=False):
        self.args = args
        self.train_loader = DataLoader(Subset(dataset, idxs), batch_size=args.batch_size, shuffle=True)
        self.device = device
        self.is_malicious = is_malicious

    # def train(self, net):
    #     net.train()
    #     optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        
    #     # 1. 保存初始权重 (用于计算 update)
    #     # 必须 clone() 和 detach()，否则会占用显存且随着计算图变化
    #     initial_params = [p.clone().detach() for p in net.parameters()]
        
    #     # 2. 初始化累积梯度列表 (用于存储 grads)
    #     grads = [] 
        
    #     for epoch in range(self.args.local_epochs):
    #         for batch_idx, (images, labels) in enumerate(self.train_loader):
    #             images, labels = images.to(self.device), labels.to(self.device)

    #             # Label Flip 攻击 (如果需要)
    #             if self.is_malicious and self.args.attack_type == "label_flip":
    #                 images, labels = perturb_data(images, labels, self.args)

    #             log_probs = net(images)
    #             loss = F.nll_loss(log_probs, labels)
                
    #             # 反向传播计算当前 batch 的梯度
    #             loss.backward()

    #             # 3. 捕获并累积原始梯度 (Raw Gradients)
    #             for i, param in enumerate(net.parameters()):
    #                 if param.requires_grad:
    #                     if len(grads) <= i:
    #                         # 如果列表还没初始化，直接追加
    #                         grads.append(param.grad.clone())
    #                     else:
    #                         # 否则累加
    #                         grads[i] += param.grad.clone()

    #             # 执行优化器更新 (改变模型权重)
    #             optimizer.step()
    #             net.zero_grad() 
        
    #     # 4. 计算模型更新量 (Weight Update / Pseudo-gradient)
    #     # update = W_final - W_initial
    #     final_params = [p for p in net.parameters()]
    #     update = [final - init for final, init in zip(final_params, initial_params)]

    #     # 5. 攻击逻辑处理 (如果 malicious)
    #     if self.is_malicious:
    #         # 你可以根据 args 选择攻击 grads 还是 update，或者都攻击
    #         # 这里假设攻击逻辑主要针对 grads 
    #         if self.args.attack_type in ["gaussian_noise", "sign_flip", "free_rider"]:
    #             grads = perturb_gradients(grads, self.args) # 注意：这里要确认 perturb 函数兼容列表格式
    #             update = [ -g * self.args.lr for g in grads ]

    #     # 6. 统一转回 CPU (可选，方便传输)
    #     net.cpu()
    #     # 将 list 里的 tensor 也转回 cpu，防止设备不匹配
    #     grads = [g.cpu() for g in grads]
    #     update = [u.cpu() for u in update]

    #     return grads, net, update

    def train(self, global_model):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()       
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        
        for _ in range(self.args.local_epochs):
            for _, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                 labels.to(device=self.args.device, non_blocking=True)

                if self.is_malicious and self.args.attack_type == "label_flip":
                    inputs, labels = perturb_data(inputs, labels, self.args)

                outputs = global_model(inputs)
                criterion = nn.CrossEntropyLoss().to(self.args.device)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                # to prevent exploding gradients
                nn.utils.clip_grad_norm_(global_model.parameters(), 10) 
                optimizer.step()
            
  
        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()) - initial_global_model_params
            grads = -update / self.args.lr

            if self.is_malicious:
                if self.args.attack_type in ["gaussian_noise", "sign_flip", "free_rider"]:
                    grads, update = perturb_gradients(grads, update, self.args)

            return grads, global_model, update
            
                # doing projected gradient descent to ensure the update is within the norm bounds 
                # if self.args.clip > 0:
                #     with torch.no_grad():
                #         local_model_params = parameters_to_vector(global_model.parameters())
                #         update = local_model_params - initial_global_model_params
                #         clip_denom = max(1, torch.norm(update, p=2)/self.args.clip)
                #         update.div_(clip_denom)
                #         vector_to_parameters(initial_global_model_params + update, global_model.parameters())
                          