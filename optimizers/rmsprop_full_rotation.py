import torch
from torch.optim.optimizer import Optimizer


class RMSpropFullRotation(Optimizer):
    r"""Implements RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, rotation_alpha=0.99, rotation_eps=1e-8,
                 lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSpropFullRotation, self).__init__(params, defaults)

        self.rotation_alpha = rotation_alpha
        self.rotation_eps = rotation_eps
        # self.rotation_momentum = rotation_momentum

        self.gamma = None
        self.rotation_state = dict()
        # TODO: make rotation_state part of self.state to support save and load

    def __setstate__(self, state):
        super(RMSpropFullRotation, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

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

            if amsgrad:     # add max to ensure convergence
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

        for p_i, p in enumerate(params):
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('RMSprop does not support sparse gradients')
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['square_avg'] = torch.zeros_like(p.data)
                if group['momentum'] > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                if group['centered']:
                    state['grad_avg'] = torch.zeros_like(p.data)

            square_avg = state['square_avg']
            alpha = group['alpha']

            state['step'] += 1

            if group['weight_decay'] != 0:
                grad = grad.add(group['weight_decay'], p.data)

            square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

            if group['centered']:
                grad_avg = state['grad_avg']
                grad_avg.mul_(alpha).add_(1 - alpha, grad)
                avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt_().add_(group['eps'])
            else:
                avg = square_avg.sqrt().add_(group['eps'])

            if group['momentum'] > 0:
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).addcdiv_(grad, avg)
                p.data.add_(-group['lr'], buf)
            else:
                p.data.addcdiv_(-group['lr'], grad, avg)

            # Update rotation
            p.data.add_(-group['lr'], add_grads[p_i])

        return loss
