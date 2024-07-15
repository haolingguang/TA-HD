import torch
from torch import autograd
import torch.nn.functional as F


class FGSM_PGN:
    def __init__(self, max_epsilon=16, norm=float('inf'), num_steps=None, args=None):

        self.eps = 2.0 * max_epsilon / 255.0
        self.num_steps = num_steps
        self.norm = norm
        self.step_alpha = self.eps / self.num_steps
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.cuda()
        self.args=args

    def non_target_attack(self, train_model, input, true_label, batch_idx=0, steps=0):

        input_var = input.clone().detach()
        label_var = true_label.clone()
        model = train_model
        step_alpha = self.step_alpha
        g = 0.0
        mu = 1.0
        steps = self.num_steps
        
        eps = self.eps
        grad = torch.zeros_like(input_var).detach().cuda()
        for step in range(steps):
            
            avg_grad = torch.zeros_like(input_var).detach().cuda()
            for _ in range(20):
                
                x_near = input_var + torch.rand_like(input_var).uniform_(-eps * 3.0, eps*3.0)
                x_near = autograd.Variable(x_near, requires_grad = True)
                output_v3 = model(x_near)
                
                loss = F.cross_entropy(output_v3, label_var)
                g1 = autograd.grad(loss, x_near,
                                            retain_graph=False, create_graph=False)[0]
                x_star = x_near.detach() + step_alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)

                nes_x = x_star.detach()
                nes_x = autograd.Variable(nes_x, requires_grad = True)
                output_v3 = model(nes_x)
                loss = F.cross_entropy(output_v3, label_var)
                g2 = autograd.grad(loss, nes_x,
                                            retain_graph=False, create_graph=False)[0]

                avg_grad += 1/20 * (0.5*g1 + 0.5 *g2)
            noise = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
            grad = mu * grad + noise
            # grad = noise
            
            input_var = input_var + step_alpha * torch.sign(grad)
            input_var = torch.clamp(input_var, -1.0, 1.0)

        return input_var.detach()