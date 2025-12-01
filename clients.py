import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.optim as optim
from attacker import perturb_data, perturb_gradients

class LocalUpdate:
    def __init__(self, args, dataset, idxs, device, is_malicious=False):
        self.args = args
        self.train_loader = DataLoader(Subset(dataset, idxs), batch_size=args.batch_size, shuffle=True)
        self.device = device
        self.is_malicious = is_malicious

    def train(self, net):
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        initial_params = [p.clone() for p in net.parameters()]
        
        for iter in range(self.args.local_epochs):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Label Flip
                if self.is_malicious and self.args.attack_type == "label_flip":
                    images, labels = perturb_data(images, labels, self.args)

                net.zero_grad()
                log_probs = net(images)
                loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
        
        # 计算梯度 (Pseudo-gradient: diff between initial and final weights)
        final_params = [p for p in net.parameters()]
        # 这里的 grads 实际上是 update vector
        grads = [final - init for final, init in zip(final_params, initial_params)]
        
        # 梯度层攻击 (Noise, Sign Flip)
        if self.is_malicious and self.args.attack_type in ["gaussian_noise", "sign_flip"]:
            grads = perturb_gradients(grads, self.args)
            
        return grads