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

        # compute RMSprop for rotation
        rotation_grad = self.gamma @ params[0].detach()      # A W1
        rotation_square_avg = self.rotation_state['square_avg']
        rotation_square_avg.mul_(self.rotation_alpha).addcmul_(1 - self.rotation_alpha, rotation_grad, rotation_grad)
        rotation_avg = rotation_square_avg.sqrt().add_(self.rotation_eps)
        rotation_grad.div_(rotation_avg)
        rotation_projection = rotation_grad @ torch.pinverse(params[0]).detach()
        rotation_projection_skew = 0.5 * (rotation_projection - rotation_projection.T)   # take skew-symmetric part

        # additional encoder and decoder grads
        add_grads = [- rotation_projection_skew @ params[0].detach(), - params[1].detach() @ rotation_projection_skew.T]

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

            # do the rotation without RMSprop
            p.data.add_(-group['lr'], add_grads[p_i])

        return loss
