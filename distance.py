import torch.nn as nn
import torch
import numpy

eps = 1e-7

class Energy_Distance(nn.Module):
    def e_dist(self, x1,x2):
        n = x1.size(0)
        m = x2.size(0)
        d = x1.size(1)

        x1 = x1.unsqueeze(1).expand(n, m, d)
        x2 = x2.unsqueeze(0).expand(n, m, d)

        #dist = torch.pow(x1 - x2, 2).sum(2)
        dist = torch.mean(torch.abs(x1 - x2).sum(2))
        return dist
    def forward(self,x1, x2):

        batches = x1.shape[0]
        e_dist = torch.zeros(batches)
        for i in range(0,batches):
            'energy distance/Crammer distance where entropy term max in Wasserstein distance'
            cross = self.e_dist(x1[i,:],x2[i,:])
            same1 = self.e_dist(x2[i,:],x2[i,:])
            same2 = self.e_dist(x1[i,:],x1[i,:])
            energy_dist = cross - 0.5*(same1 + same2)
            e_dist[i] = energy_dist
        return e_dist


class KLDiv(nn.Module):
    # Calculate KL-Divergence

    def forward(self, predict, target):
        assert predict.ndimension() == 2, 'Input dimension must be 2'
        target = target.detach()

        # KL(T||I) = \sum T(logT-logI)
        predict = eps + predict
        target = eps + target

        logI = torch.log(predict)
        logT = torch.log(target)
        logdiff = logT - logI
        TlogTdI = target * (logdiff)
        kld = TlogTdI.sum(1)
        #  criter = nn.MSELoss()
        #  kld = criter(predict,target)

        return kld


class Gaussian_KL(nn.Module):
    def forward(self,X1, X2):
        'assumes independence. Also assumes B x dimension in shape'
        #mu1 = torch.mean(signal1,dim=1)
        #mu2 = torch.mean(signal2, dim =1)

        #var1 = torch.var(signal1, dim =1)
        #var2 = torch.var(signal2, dim =1)

        mu1 = X1[0]
        var1 = X1[1]

        mu2 = X2[0]
        var2 = X2[1]
        var1 = var1 +1e-6
        var2 = var2+1e-6

        #assumes independence




        var2_inv = var2**(-1)

        det_var1 = 0
        det_var2 = 0

        if var1.shape[1] > 1:
            for i in range(0, var1.shape[1] - 1):
                det_var1 = det_var1 + var1[:, i] * var1[:, i + 1]
                det_var2 = det_var2 + var2[:, i] * var2[:, i + 1]
        else:
            det_var1 = var1
            det_var2 = var2

        det_var1 = det_var1.reshape(-1)
        det_var2 = det_var2.reshape(-1)

        kl = 0.5 *  (torch.log(det_var2) - torch.log( det_var1) - var1.shape[1] + torch.sum(var2_inv* var1,dim=1)+ torch.sum(
           (mu1 - mu2) * var2_inv * (mu1 - mu2) , dim = 1)).reshape(-1,1)

        #kl2 = torch.log(torch.sqrt(var2)) - torch.log(torch.sqrt(var1)) + (var1 +(mu1 - mu2)**2)/(2*var2) - 0.5
        return  kl


class Window_gauss_KL(nn.Module):
    def forward(self,signal1, signal2):
        mu1 = torch.mean(signal1,dim=1)
        mu2 = torch.mean(signal2, dim =1)

        var1 = torch.var(signal1, dim =1) + 1e-6
        var2 = torch.var(signal2, dim =1)  +1e-6

        var2_inv = var2 ** (-1)

        det_var1 = 0
        det_var2 = 0

        if var1.shape[1] > 1:
            for i in range(0, var1.shape[1] - 1):
                det_var1 = det_var1 + var1[:, i] * var1[:, i + 1]
                det_var2 = det_var2 + var2[:, i] * var2[:, i + 1]
        else:
            det_var1 = var1
            det_var2 = var2

        det_var1 = det_var1.reshape(-1)
        det_var2 = det_var2.reshape(-1)

        kl = 0.5 * (torch.log(det_var2) - torch.log(det_var1) - var1.shape[1] + torch.sum(var2_inv * var1,
                                                                                          dim=1) + torch.sum(
            (mu1 - mu2) * var2_inv * (mu1 - mu2), dim=1)).reshape(-1, 1)

        return  kl


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :


        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        C = self._cost_matrix(x, y)  # Wasserstein cost function

        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()


        mu = mu.cuda()
        nu = nu.cuda()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        #return cost, pi, C
        return cost
    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"

        u.cuda()
        v.cuda()
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class SinkhornDistance_Pytorch(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps =1, max_iter=200, reduction='none'):
        super(SinkhornDistance_Pytorch, self).__init__()
        self.sk_dist = SinkhornDistance(eps= eps, max_iter=max_iter)

    def forward(self, x, y):
        sk_dist = self.sk_dist(x, y) - 0.5 * self.sk_dist(x, x) - 0.5 * self.sk_dist(y, y)
        return sk_dist

class SinkhornDistance_Mahal(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance_Mahal, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y,M):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y,M)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        mu = mu.cuda()
        nu = nu.cuda()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C
       # return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"

        u.cuda()
        v.cuda()
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, M,p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        #x_col = x.unsqueeze(-2)
        #y_lin = y.unsqueeze(0)
        #M = M.float()
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        win_length = x.shape[1]
        #x_temp = x_col.repeat(1, win_length, 1)
        #y_temp = y_lin.repeat(win_length,1,1)

        #x_temp = x_col.repeat(1, win_length, 1)
        #y_temp = y_lin.repeat(win_length,1,1)

        #C = torch.sum(torch.sum((torch.abs(x_col - y_lin)) ** p, -1), axis = 0)





        C_mahal = torch.sum(((x_col - y_lin) @ M ) * (x_col - y_lin),axis = -1)
        return C_mahal

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1