"""
Adam-HN-AC: Adam with Hybrid Norm and Adaptive Curvature
=======================================================

A novel optimization algorithm for deep neural networks that extends
HN_Adam with adaptive curvature estimation and dynamic momentum adjustment.

Based on:
- Kingma & Ba (2015): Adam: A method for stochastic optimization
- Reyad et al. (2023): A modified Adam algorithm for deep neural network optimization

Author: Research Implementation
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
import copy


# ============================================================
# PROPOSED OPTIMIZER: Adam-HN-AC
# ============================================================

class Adam_HN_AC(torch.optim.Optimizer):
    """
    Adam-HN-AC (Adam with Hybrid Norm and Adaptive Curvature)
    
    A novel optimization algorithm that combines:
    1. Adaptive curvature estimation using gradient history
    2. Dynamic momentum adjustment based on loss landscape
    3. Enhanced AMSGrad mechanism for stable second-moment estimation
    4. Hybrid norm regularization for adaptive learning rate
    
    Parameters:
    -----------
    params : iterable
        Iterable of parameters to optimize
    lr : float, default=0.001
        Learning rate
    betas : tuple, default=(0.9, 0.999)
        Coefficients for running averages of gradient and its square
    eps : float, default=1e-8
        Term added for numerical stability
    weight_decay : float, default=0
        Weight decay (L2 penalty)
    alpha : float, default=0.1
        Hybrid norm coefficient for adaptive learning rate
    curvature_weight : float, default=0.01
        Weight for curvature influence on momentum
    noise_factor : float, default=0.0
        Gradient noise factor for regularization (optional)
    
    References:
    -----------
    - Kingma & Ba (2015): Adam: A method for stochastic optimization (ICLR)
    - Reyad et al. (2023): Neural Computing and Applications
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, alpha=0.1, curvature_weight=0.01, noise_factor=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                       alpha=alpha, curvature_weight=curvature_weight, noise_factor=noise_factor)
        super(Adam_HN_AC, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Parameters:
        -----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.
        
        Returns:
        --------
        loss : float
            The loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam_HN_AC does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prev_update'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)
                    state['curvature'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                max_exp_avg_sq = state['max_exp_avg_sq']
                prev_update = state['prev_update']
                prev_grad = state['prev_grad']
                curvature = state['curvature']
                beta1, beta2 = group['betas']
                alpha = group['alpha']
                curvature_weight = group['curvature_weight']
                noise_factor = group['noise_factor']
                
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # ============================================================
                # ADAPTIVE CURVATURE ESTIMATION
                # ============================================================
                if state['step'] > 1:
                    # Compute gradient difference for curvature estimation
                    grad_diff = grad - prev_grad
                    curvature.mul_(0.9).add_(grad_diff.abs(), alpha=0.1)
                    
                    # Adaptive momentum based on curvature
                    # Lower momentum in high-curvature regions
                    adaptive_beta1 = beta1 * (1 - curvature_weight * torch.tanh(curvature.mean()))
                else:
                    adaptive_beta1 = beta1
                
                # Store current gradient for next iteration
                prev_grad.copy_(grad)
                
                # Add gradient noise for better generalization (optional)
                if noise_factor > 0 and state['step'] > 100:
                    noise = torch.randn_like(grad) * noise_factor * grad.std()
                    grad = grad + noise
                
                # ============================================================
                # FIRST MOMENT UPDATE (with adaptive beta1)
                # ============================================================
                exp_avg.mul_(adaptive_beta1).add_(grad, alpha=1 - adaptive_beta1)
                
                # ============================================================
                # SECOND MOMENT UPDATE (AMSGrad mechanism)
                # ============================================================
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # AMSGrad: maintain maximum of second moment estimates
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                
                # ============================================================
                # BIAS CORRECTION
                # ============================================================
                bias_correction1 = 1 - adaptive_beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # ============================================================
                # ADAPTIVE LEARNING RATE (Hybrid Norm mechanism)
                # ============================================================
                if state['step'] > 1:
                    update_norm = prev_update.norm()
                    adaptive_lr = group['lr'] * (1 + alpha * torch.tanh(update_norm))
                else:
                    adaptive_lr = group['lr']
                
                # Compute step size with bias correction
                step_size = adaptive_lr / bias_correction1
                
                # Compute denominator with AMSGrad
                denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                
                # ============================================================
                # PARAMETER UPDATE
                # ============================================================
                update = exp_avg / denom
                
                # Store update for next iteration's adaptive learning rate
                prev_update.copy_(update)
                
                # Apply update
                p.data.add_(update, alpha=-step_size)
        
        return loss


# ============================================================
# BASELINE OPTIMIZERS FOR COMPARISON
# ============================================================

class AdamOptimizer(torch.optim.Optimizer):
    """
    Standard Adam Optimizer (baseline for comparison)
    Kingma & Ba (2015): Adam: A method for stochastic optimization
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step size
                step_size = group['lr'] / bias_correction1
                
                # Compute denominator
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        
        return loss


class HN_Adam(torch.optim.Optimizer):
    """
    HN_Adam: Hybrid Norm Adam Optimizer
    Based on: Reyad et al. (2023) - Neural Computing and Applications
    
    Key improvements over standard Adam:
    - Automatically adjusts step size based on parameter update norm
    - Combines Adam and AMSGrad algorithms
    - Better generalization performance
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, alpha=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, alpha=alpha)
        super(HN_Adam, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('HN_Adam does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prev_update'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                max_exp_avg_sq = state['max_exp_avg_sq']
                prev_update = state['prev_update']
                beta1, beta2 = group['betas']
                alpha = group['alpha']
                
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # AMSGrad: maintains the maximum of all second moment estimates
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute adaptive step size based on norm of previous update
                if state['step'] > 1:
                    update_norm = prev_update.norm()
                    adaptive_lr = group['lr'] * (1 + alpha * torch.tanh(update_norm))
                else:
                    adaptive_lr = group['lr']
                
                # Compute step size with bias correction
                step_size = adaptive_lr / bias_correction1
                
                # Compute denominator using max_exp_avg_sq (AMSGrad mechanism)
                denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                
                # Compute update
                update = exp_avg / denom
                
                # Store update for next iteration
                prev_update.copy_(update)
                
                # Update parameters
                p.data.add_(update, alpha=-step_size)
        
        return loss


# ============================================================
# CNN MODEL ARCHITECTURES
# ============================================================

class CNN_MNIST(nn.Module):
    """
    Convolutional Neural Network for MNIST Dataset
    
    Architecture:
    -----------
    Conv(32) → BatchNorm → ReLU → MaxPool → 
    Conv(64) → BatchNorm → ReLU → MaxPool → 
    Flatten → FC(128) → Dropout → FC(10)
    
    Parameters:
    -----------
    None (architecture is fixed)
    
    Input:
    ------
    - 28×28 grayscale images
    
    Output:
    -------
    - 10 class logits (digits 0-9)
    """
    
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNN_CIFAR10(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 Dataset
    
    Architecture:
    -----------
    VGG-style architecture with batch normalization
    3 blocks of Conv → Conv → MaxPool with increasing channels
    
    Parameters:
    -----------
    None (architecture is fixed)
    
    Input:
    ------
    - 32×32 RGB images
    
    Output:
    -------
    - 10 class logits
    """
    
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch
    
    Parameters:
    -----------
    model : nn.Module
        Neural network model
    train_loader : DataLoader
        Training data loader
    optimizer : Optimizer
        Optimization algorithm
    criterion : Loss
        Loss function
    device : torch.device
        Device to run on
    
    Returns:
    --------
    epoch_loss : float
        Average loss for the epoch
    epoch_acc : float
        Accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model
    
    Parameters:
    -----------
    model : nn.Module
        Neural network model
    test_loader : DataLoader
        Test data loader
    criterion : Loss
        Loss function
    device : torch.device
        Device to run on
    
    Returns:
    --------
    test_loss : float
        Average test loss
    test_acc : float
        Test accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


def train_model(model, train_loader, test_loader, optimizer, criterion, 
                epochs, device, verbose=True):
    """
    Complete training loop with history tracking
    
    Parameters:
    -----------
    model : nn.Module
        Neural network model
    train_loader : DataLoader
        Training data loader
    test_loader : DataLoader
        Test data loader
    optimizer : Optimizer
        Optimization algorithm
    criterion : Loss
        Loss function
    epochs : int
        Number of training epochs
    device : torch.device
        Device to run on
    verbose : bool
        Print progress
    
    Returns:
    --------
    history : dict
        Training history containing losses and accuracies
    best_acc : float
        Best test accuracy achieved
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_time': []
    }
    
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = copy.deepcopy(model.state_dict())
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
                  f'Time: {epoch_time:.2f}s')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history, best_acc


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example: Create model and optimizer
    model = CNN_MNIST().to(device)
    optimizer = Adam_HN_AC(model.parameters(), lr=0.001, alpha=0.1, curvature_weight=0.01)
    
    print("\nModel Architecture:")
    print(model)
    
    print("\nOptimizer Configuration:")
    print(f"  Learning Rate: {optimizer.defaults['lr']}")
    print(f"  Beta1: {optimizer.defaults['betas'][0]}")
    print(f"  Beta2: {optimizer.defaults['betas'][1]}")
    print(f"  Alpha (Hybrid Norm): {optimizer.defaults['alpha']}")
    print(f"  Curvature Weight: {optimizer.defaults['curvature_weight']}")
    
    print("\nAdam-HN-AC optimizer ready for training!")
