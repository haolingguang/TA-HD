import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F
import torch_dct as dct


class FGSM_SIA:
    def __init__(self, max_epsilon=16, norm=float('inf'), num_steps=None):

        self.eps = 2.0 * max_epsilon / 255.0
        self.num_steps = num_steps
        self.norm = norm
        self.step_alpha = self.eps / self.num_steps
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.cuda()
        self.SIA_transform = SIA(20, 3)

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
        label_var = label_var.repeat(20)
        
        while step < steps:
            input_var.grad = None
            output = model(self.SIA_transform(input_var))
            loss = self.loss_fn(output, label_var)
            loss.backward()

            if self.norm == 2:
                normed_grad = torch.norm(torch.flatten(input_var.grad.data, start_dim=1), p=2 ,dim=1)
                normed_grad = torch.flatten(input_var.grad.data, start_dim=1)/ normed_grad.reshape(input_var.shape[0],-1)
                normed_grad = normed_grad.reshape(input_var.shape)
            elif self.norm == 1:
                normed_grad = torch.norm(torch.flatten(input_var.grad.data, start_dim=1), p=1 ,dim=1)
                normed_grad = torch.flatten(input_var.grad.data, start_dim=1)/ normed_grad.reshape(input_var.shape[0],-1)
                normed_grad = normed_grad.reshape(input_var.shape)
            else:
                # infinity-norm
                normed_grad = torch.sign(input_var.grad.data)
            
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
     

class SIA(torch.nn.Module):

    def __init__(self, num_copies=20, num_block=3):
        super(SIA,self).__init__()
        self.num_copies = num_copies
        self.num_block = num_block
        self.op = [self.resize, self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.rotate180, self.scale, self.add_noise,self.dct,self.drop_out]
        

    def forward(self, x):
        return torch.cat([self.blocktransform(x) for _ in range(self.num_copies)])
    
        
    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low = 0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)


    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low = 0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)


    def vertical_flip(self, x):
        return x.flip(dims=(2,))


    def horizontal_flip(self, x):
        return x.flip(dims=(3,))


    def rotate180(self, x):
        return x.rot90(k=2, dims=(2,3))
    
    
    def scale(self, x):
        return torch.rand(1)[0] * x
    
    
    def resize(self, x):
        """
        Resize the input
        """
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor)+1
        new_w = int(w * scale_factor)+1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
        return x
    
    
    def dct(self, x):
        """
        Discrete Fourier Transform
        """
        dctx = dct.dct_2d(x) #torch.fft.fft2(x, dim=(-2, -1))
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        # dctx[:, :, -low_w:, -low_h:] = 0
        dctx[:, :, -low_w:,:] = 0
        dctx[:, :, :, -low_h:] = 0
        dctx = dctx # * self.mask.reshape(1, 1, w, h)
        idctx = dct.idct_2d(dctx)
        return idctx
    
    
    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), 0, 1)


    def drop_out(self, x):
        return F.dropout2d(x, p=0.1, training=True)


    def blocktransform(self, x, choice=-1):
        _, _, w, h = x.shape
        y_axis = [0,] + np.random.choice(list(range(1, h)), self.num_block-1, replace=False).tolist() + [h,]
        x_axis = [0,] + np.random.choice(list(range(1, w)), self.num_block-1, replace=False).tolist() + [w,]
        y_axis.sort()
        x_axis.sort()
        
        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])
        return x_copy



