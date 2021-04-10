import torch
from torch.optim.optimizer import Optimizer


class RMSpropRotation(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, rotation_alpha=0.99, rotation_eps=1e-8,
                 lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(RMSpropRotation, self).__init__(params, defaults)

        self.rotation_alpha = rotation_alpha
        self.rotation_eps = rotation_eps
        # self.rotation_momentum = rotation_momentum

        self.gamma = None
        self.rotation_state = dict()
        # TODO: make rotation_state part of self.state to support save and load

    def __setstate__(self, state):
        super(RMSpropRotation, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def record_current_rotation(self, gamma):
        self.gamma = gamma.detach()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # only one parameter group
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params = group['params']

        # 2 parameter tensors (encoder and decoder)
        assert len(params) == 2

        # rotation state initialization
        if len(self.rotation_state) == 0:
            self.rotation_state['square_avg'] = torch.zeros_like(params[0].data).detach()
            self.rotation_state['momentum_buffer'] = torch.zeros_like(params[0].data).detach()
            self.rotation_state['diag_square_avg_r'] = torch.zeros((params[0].shape[0],)).detach().cuda()

        # compute RMSprop for rotation
        rotation_grad = self.gamma @ params[0].detach()  # A W1

        row_space_adapt = True
        amsgrad = False
        if row_space_adapt:
            diag_rotation_square_avg_r = self.rotation_state['diag_square_avg_r']

            # column space cov (row space of AW)
            rotation_sq = rotation_grad @ rotation_grad.T / float(params[0].shape[1])

            # EMA on diag covariance
            diag_rotation_square_avg_r.mul_(self.rotation_alpha).add_(1 - self.rotation_alpha, torch.diag(rotation_sq))

            if amsgrad:  # add max to ensure convergence
                diag_rotation_square_avg_r_copy = diag_rotation_square_avg_r.clone()
                max_diag_r = torch.max(diag_rotation_square_avg_r, diag_rotation_square_avg_r_copy)
                rotation_grad = torch.diag(torch.pow(max_diag_r, - 0.5)) @ rotation_grad
            else:
                rotation_grad = torch.diag(torch.pow(diag_rotation_square_avg_r, - 0.5)) @ rotation_grad

        else:  # element-wise adaptation (naive)
            rotation_square_avg = self.rotation_state['square_avg']

            canonical_inner_prod = False
            if canonical_inner_prod:
                raise NotImplementedError
            else:
                rotation_sq = torch.mul(rotation_grad, rotation_grad)

            rotation_square_avg.mul_(self.rotation_alpha).add_(1 - self.rotation_alpha, rotation_sq)
            rotation_avg = rotation_square_avg.sqrt().add_(self.rotation_eps)
            rotation_grad.div_(rotation_avg)

        # # momentum  # TODO: add transport
        # rotation_grad_buf = self.rotation_state['momentum_buffer']
        # rotation_grad_buf.mul_(self.rotation_momentum).add_(rotation_grad)
        # rotation_grad = rotation_grad_buf

        # ==== project to tangent space ====
        # ---- Stiefel manifold projection to tangent space (transpose instead of pinverse) ----
        z_wt = rotation_grad @ params[0].T.detach()
        skew_z_wt = 0.5 * (z_wt - z_wt.T)
        add_grads = [- skew_z_wt @ params[0].detach(),
                     - params[1].detach() @ skew_z_wt.T]
        # -------------------------
        # =========================

        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p_i, p in enumerate(params):
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf

            p.data.add_(-group['lr'], d_p)

            # Update rotation
            p.data.add_(-group['lr'], add_grads[p_i])

        return loss
