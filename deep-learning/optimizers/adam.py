"""
Adam and AdamW Optimizers
Implementation from scratch with bias correction
"""

import torch
import math
from typing import Tuple


class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer

    Implements the update rule:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        m̂_t = m_t / (1 - β1^t)
        v̂_t = v_t / (1 - β2^t)
        θ_{t+1} = θ_t - lr * m̂_t / (√v̂_t + ε)

    Where:
        θ = parameters
        g = gradients
        m = first moment estimate (mean)
        v = second moment estimate (uncentered variance)
        β1, β2 = exponential decay rates
        lr = learning rate
        ε = small constant for numerical stability
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False
    ):
        """
        Initialize Adam optimizer

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay (L2 penalty)
            amsgrad: Whether to use AMSGrad variant
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        # First moment estimate (mean)
        self.m = [torch.zeros_like(p) for p in self.params]

        # Second moment estimate (uncentered variance)
        self.v = [torch.zeros_like(p) for p in self.params]

        # Maximum of second moment estimate (for AMSGrad)
        if self.amsgrad:
            self.v_max = [torch.zeros_like(p) for p in self.params]

        # State
        self.state = {
            'step': 0
        }

    def zero_grad(self):
        """Zero out the gradients of all parameters"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        """
        Perform a single optimization step
        Updates parameters based on their gradients
        """
        self.state['step'] += 1
        t = self.state['step']

        # Bias correction terms
        bias_correction1 = 1 - self.beta1 ** t
        bias_correction2 = 1 - self.beta2 ** t

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data

            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad.add(param.data, alpha=self.weight_decay)

            # Update biased first moment estimate
            self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)

            # Update biased second raw moment estimate
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            if self.amsgrad:
                # Maintain max of all 2nd moment running avg till now
                torch.max(self.v_max[i], self.v[i], out=self.v_max[i])

                # Use max for normalizing running avg of gradient
                denom = (self.v_max[i].sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            else:
                denom = (self.v[i].sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

            # Compute step size
            step_size = self.lr / bias_correction1

            # Update parameters
            param.data.addcdiv_(self.m[i], denom, value=-step_size)

    def state_dict(self):
        """Return optimizer state as dictionary"""
        state_dict = {
            'state': self.state,
            'param_groups': [{
                'lr': self.lr,
                'betas': (self.beta1, self.beta2),
                'eps': self.eps,
                'weight_decay': self.weight_decay,
                'amsgrad': self.amsgrad
            }],
            'm': self.m,
            'v': self.v
        }

        if self.amsgrad:
            state_dict['v_max'] = self.v_max

        return state_dict

    def load_state_dict(self, state_dict):
        """Load optimizer state from dictionary"""
        self.state = state_dict['state']
        param_group = state_dict['param_groups'][0]
        self.lr = param_group['lr']
        self.beta1, self.beta2 = param_group['betas']
        self.eps = param_group['eps']
        self.weight_decay = param_group['weight_decay']
        self.amsgrad = param_group['amsgrad']
        self.m = state_dict['m']
        self.v = state_dict['v']

        if self.amsgrad:
            self.v_max = state_dict['v_max']


class AdamW:
    """
    AdamW optimizer (Adam with decoupled weight decay)

    Unlike Adam, weight decay is decoupled from gradient-based update.
    This provides better regularization and generalization.

    Update rule:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        m̂_t = m_t / (1 - β1^t)
        v̂_t = v_t / (1 - β2^t)
        θ_{t+1} = θ_t - lr * (m̂_t / (√v̂_t + ε) + λ * θ_t)

    Where λ is the weight decay coefficient.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ):
        """
        Initialize AdamW optimizer

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay coefficient (decoupled)
            amsgrad: Whether to use AMSGrad variant
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        # First moment estimate
        self.m = [torch.zeros_like(p) for p in self.params]

        # Second moment estimate
        self.v = [torch.zeros_like(p) for p in self.params]

        # Maximum of second moment estimate (for AMSGrad)
        if self.amsgrad:
            self.v_max = [torch.zeros_like(p) for p in self.params]

        # State
        self.state = {
            'step': 0
        }

    def zero_grad(self):
        """Zero out the gradients of all parameters"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        """
        Perform a single optimization step with decoupled weight decay
        """
        self.state['step'] += 1
        t = self.state['step']

        # Bias correction terms
        bias_correction1 = 1 - self.beta1 ** t
        bias_correction2 = 1 - self.beta2 ** t

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data

            # Update biased first moment estimate
            self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)

            # Update biased second raw moment estimate
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            if self.amsgrad:
                # Maintain max of all 2nd moment running avg till now
                torch.max(self.v_max[i], self.v[i], out=self.v_max[i])

                # Use max for normalizing running avg of gradient
                denom = (self.v_max[i].sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            else:
                denom = (self.v[i].sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

            # Compute step size
            step_size = self.lr / bias_correction1

            # AdamW: Decoupled weight decay
            # Apply weight decay directly to parameters (not to gradients)
            if self.weight_decay != 0:
                param.data.mul_(1 - self.lr * self.weight_decay)

            # Update parameters
            param.data.addcdiv_(self.m[i], denom, value=-step_size)

    def state_dict(self):
        """Return optimizer state as dictionary"""
        state_dict = {
            'state': self.state,
            'param_groups': [{
                'lr': self.lr,
                'betas': (self.beta1, self.beta2),
                'eps': self.eps,
                'weight_decay': self.weight_decay,
                'amsgrad': self.amsgrad
            }],
            'm': self.m,
            'v': self.v
        }

        if self.amsgrad:
            state_dict['v_max'] = self.v_max

        return state_dict

    def load_state_dict(self, state_dict):
        """Load optimizer state from dictionary"""
        self.state = state_dict['state']
        param_group = state_dict['param_groups'][0]
        self.lr = param_group['lr']
        self.beta1, self.beta2 = param_group['betas']
        self.eps = param_group['eps']
        self.weight_decay = param_group['weight_decay']
        self.amsgrad = param_group['amsgrad']
        self.m = state_dict['m']
        self.v = state_dict['v']

        if self.amsgrad:
            self.v_max = state_dict['v_max']


# Example usage and comparison
if __name__ == "__main__":
    import torch.nn as nn

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

    print("Testing Adam optimizer:")
    print("-" * 50)

    # Create Adam optimizer
    optimizer_adam = Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # Dummy training loop
    for epoch in range(5):
        x = torch.randn(32, 10)
        y = torch.randint(0, 10, (32,))

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)

        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    print("\n" + "=" * 50)
    print("Testing AdamW optimizer:")
    print("-" * 50)

    # Reset model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

    # Create AdamW optimizer
    optimizer_adamw = AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Dummy training loop
    for epoch in range(5):
        x = torch.randn(32, 10)
        y = torch.randint(0, 10, (32,))

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)

        optimizer_adamw.zero_grad()
        loss.backward()
        optimizer_adamw.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    print("\nAdam and AdamW optimizers tested successfully!")
