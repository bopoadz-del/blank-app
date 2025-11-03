"""
Stochastic Gradient Descent (SGD) Optimizer
Implementation from scratch with momentum and weight decay
"""

import torch
from typing import List, Optional


class SGD:
    """
    Stochastic Gradient Descent optimizer with momentum

    Implements the update rule:
        v_{t+1} = momentum * v_t + g_t
        θ_{t+1} = θ_t - lr * v_{t+1}

    Where:
        θ = parameters
        g = gradients
        v = velocity (momentum buffer)
        lr = learning rate
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False
    ):
        """
        Initialize SGD optimizer

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            momentum: Momentum factor (0 = no momentum)
            weight_decay: Weight decay (L2 penalty)
            dampening: Dampening for momentum
            nesterov: Enable Nesterov momentum
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

        # Momentum buffers
        self.velocity = [torch.zeros_like(p) for p in self.params]

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

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data

            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad.add(param.data, alpha=self.weight_decay)

            # Apply momentum
            if self.momentum != 0:
                velocity = self.velocity[i]

                if self.state['step'] == 1:
                    # First step: initialize velocity
                    velocity.copy_(grad)
                else:
                    # Update velocity
                    velocity.mul_(self.momentum).add_(grad, alpha=1 - self.dampening)

                if self.nesterov:
                    # Nesterov momentum
                    grad = grad.add(velocity, alpha=self.momentum)
                else:
                    # Standard momentum
                    grad = velocity

            # Update parameters
            param.data.add_(grad, alpha=-self.lr)

    def state_dict(self):
        """Return optimizer state as dictionary"""
        return {
            'state': self.state,
            'param_groups': [{
                'lr': self.lr,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay,
                'dampening': self.dampening,
                'nesterov': self.nesterov
            }],
            'velocity': self.velocity
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state from dictionary"""
        self.state = state_dict['state']
        param_group = state_dict['param_groups'][0]
        self.lr = param_group['lr']
        self.momentum = param_group['momentum']
        self.weight_decay = param_group['weight_decay']
        self.dampening = param_group['dampening']
        self.nesterov = param_group['nesterov']
        self.velocity = state_dict['velocity']


class SGDWithLRSchedule(SGD):
    """
    SGD with learning rate scheduling
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        lr_decay: float = 0.1,
        lr_decay_epochs: Optional[List[int]] = None
    ):
        """
        Initialize SGD with learning rate schedule

        Args:
            params: Parameters to optimize
            lr: Initial learning rate
            momentum: Momentum factor
            weight_decay: Weight decay
            dampening: Dampening for momentum
            nesterov: Enable Nesterov momentum
            lr_decay: Learning rate decay factor
            lr_decay_epochs: Epochs at which to decay learning rate
        """
        super().__init__(params, lr, momentum, weight_decay, dampening, nesterov)

        self.initial_lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_epochs = lr_decay_epochs or []

    def step_epoch(self, epoch: int):
        """
        Adjust learning rate based on epoch

        Args:
            epoch: Current epoch number
        """
        if epoch in self.lr_decay_epochs:
            self.lr *= self.lr_decay
            print(f"Epoch {epoch}: Learning rate decayed to {self.lr:.6f}")


# Example usage
if __name__ == "__main__":
    import torch.nn as nn

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

    # Create SGD optimizer
    optimizer = SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )

    # Dummy training loop
    for epoch in range(5):
        # Forward pass
        x = torch.randn(32, 10)
        y = torch.randint(0, 10, (32,))

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimization step
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    print("\nSGD optimizer test completed successfully!")
