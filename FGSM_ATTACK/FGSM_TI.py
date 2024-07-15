import numpy as np
import torch
from torch import autograd
import scipy.stats as st
import torch.nn.functional as F


class FGSM_TI:
    def __init__(self, max_epsilon=16, norm=float('inf'), num_steps=None):

        self.eps = 2.0 * max_epsilon / 255.0
        self.num_steps = num_steps
        self.norm = norm
        self.step_alpha = self.eps / self.num_steps
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.cuda()
        self.gauss_kernel = torch.FloatTensor(gkern(args.kernel_size,3)).cuda()

    def non_target_attack(self, train_model, input, true_label, batch_idx=0, steps=0):

        input_var = autograd.Variable(input, requires_grad=True)
        label_var = autograd.Variable(true_label)
        model = train_model
        step_alpha = self.step_alpha
        g = 0.0
        mu = 1.0
        steps = self.num_steps
        step = 0
        eps =self.eps
        while step < steps:
            input_var.grad = None
            output = model(input_var)
            loss = self.loss_fn(output, label_var)
            loss.backward()
            
            grad=F.conv2d(input_var.grad.data, self.gauss_kernel, stride=1, padding=args.kernel_size//2)
            if self.norm == 2:
                normed_grad = torch.norm(torch.flatten(grad, start_dim=1), p=2 ,dim=1)
                normed_grad = torch.flatten(grad, start_dim=1)/ normed_grad.reshape(input_var.shape[0],-1)
                normed_grad = normed_grad.reshape(input_var.shape)
            elif self.norm == 1:
                normed_grad = torch.norm(torch.flatten(grad, start_dim=1), p=1 ,dim=1)
                normed_grad = torch.flatten(grad, start_dim=1)/ normed_grad.reshape(input_var.shape[0],-1)
                normed_grad = normed_grad.reshape(input_var.shape)
            else:
                # infinity-norm
                normed_grad = torch.sign(grad)
            
            g = mu*g + normed_grad
            step_per = input_var.data + step_alpha * torch.sign(g)
            
            # calculate total adversarial perturbation from original image and clip to epsilon constraints
            total_per = step_per - input
            total_per = torch.clamp(total_per, -eps, eps)
            # apply total adversarial perturbation to original image and clip to valid pixel range
            input_adv = input + total_per
            input_adv = torch.clamp(input_adv, -1.0, 1.0)
            input_var.data = input_adv
            step += 1
        return input_adv    
     

def gkern(kernlen=21, nsig=3):
    
    noise = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(noise)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel,0)
    stack_kernel = np.repeat(stack_kernel, 3, axis=0)
    return stack_kernel



