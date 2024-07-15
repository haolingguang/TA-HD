import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch_dct as dct

class HD_SIA:
    def __init__(self, max_epsilon, norm, num_steps, T_net, T_optimizer):
        '''initialize the parameters'''
         
        self.eps = 2.0 * max_epsilon/255.0
        self.norm = norm
        self.num_steps = num_steps
        self.T_net = T_net
        self.T_optimizer = T_optimizer
        self.step_alpha = self.eps / self.num_steps
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_T = nn.MSELoss(reduction='mean')
        self.num_copies = 20
        self.num_block = 3
        self.SIA_transform = SIA(self.num_copies, self.num_block)
        
        
    def train_T(self, train_model, input, true_label, writer, loss_mode, T_Net_name, global_step):
        '''Train HD'''
        
        input_clear = torch.autograd.Variable(input, requires_grad=False)
        input_var = torch.autograd.Variable(input, requires_grad=True)
        label_var = torch.autograd.Variable(true_label)
        model = train_model
        step_alpha = self.step_alpha
        g = 0.0
        mu = 1.0
        steps = self.num_steps
        step = 0
        eps =self.eps
        label_var = label_var.repeat(self.num_copies)
        
        while step < steps:
            input_var.grad = None
            self.T_optimizer.zero_grad()
            #train adversarial example
            output_f1 = model(self.SIA_transform(input_var))
            output_f2 = model(self.T_net(self.SIA_transform(input_var)))
            # output_f1 = model(self.T_net(input_var))
            # output_f2 = model(input_var)  
            loss_f1 = self.loss_fn(output_f1, label_var)
            loss_f2 = self.loss_fn(output_f2, label_var)
            loss_fool = loss_f1 + loss_f2
            loss_fool.backward()

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
            step_adv = input_var.data + step_alpha * torch.sign(g)
            # calculate total adversarial perturbation from original image and clip to epsilon constraints
            total_per = step_adv - input
            total_per = torch.clamp(total_per, -eps, eps)
            # apply total adversarial perturbation to original image and clip to valid pixel range
            input_adv = input + total_per
            input_adv = torch.clamp(input_adv, -1.0, 1.0)
            input_var.data = input_adv
            step += 1

        input_var.grad = None
        input_clear.grad = None

        self.T_optimizer.zero_grad()
        output_T1 = model(self.T_net(input_var))
        output_T2 = model(self.T_net(input_clear))
        loss_T1 = self.loss_fn(output_T1, true_label)
        loss_T2 = self.loss_fn(output_T2, true_label)
        
        x = self.T_net(input_var)
        if T_Net_name == "Conv_Net":
            loss_T3 = input_var-x   # 这里是原来的ATTA论文中吴卫兵用的loss，期望这个T_Net不要矫正过剩
        else:
            loss_T3 = input_clear-x  # 这个默认是我们提出的方法中的loss，即降噪后的样本和清洁样本靠近
        
        loss_T3 = torch.mean(torch.norm(torch.flatten(loss_T3, start_dim=1), p=2 ,dim=1))
        # loss_T4 = self.loss_T(output_T1,output_T2)  # #这个是CGD loss
        # MSE_loss = self.loss_T(x,input_var)
        # loss_T4 = metrics.mutual_info_score(self.T_net(input_var), input_clear)
        
        if loss_mode=='CGD':
            loss_T = loss_T1 + loss_T2
        elif loss_mode=='LGD_CGD':
            loss_T = loss_T1 + loss_T2 + loss_T3
        # elif loss_mode=='CGD_HGD':
        #     loss_T = loss_T1 + loss_T2 + loss_T3 + loss_T4      
        # elif loss_mode=='LGD_MI':
        #     loss_T = loss_T3 - loss_T4
        # elif loss_mode=='CGD_LGD_MI':
        #     loss_T = loss_T1 + loss_T2 + loss_T3-loss_T4
        
        
        
        # loss_T = loss_T1 + loss_T2 + loss_T3 - loss_T4
        # loss_T = loss_T1 + loss_T2 - loss_T4
        # loss_T = loss_T1 + loss_T2
        # loss_T = self.loss_T(output_T1,output_T2)
        
        # 可视化loss
        writer.add_scalar('Loss/Loss_T1', loss_T1, global_step)
        writer.add_scalar('Loss/Loss_T2', loss_T2, global_step)
        writer.add_scalar('Loss/Loss_T3', loss_T3, global_step)
        # writer.add_scalar('Loss/Loss_T4', loss_T4, global_step)
        # writer.add_scalar('Loss/Loss_T', loss_T, global_step)
        
        # 可视化图片
        # grid = make_grid(input_var/2+1)
        # grid = make_grid(input_clear/2+1)
        # writer.add_image('Clean/images', grid)
        # writer.add_image('Adversarial/images', grid)
        
        loss_T.backward()
        self.T_optimizer.step()

        # return input_adv
    def test_T(self, train_model, input, true_label):
        '''Test the performance of HD'''
        
        input_var = torch.autograd.Variable(input, requires_grad=True)
        label_var = torch.autograd.Variable(true_label)
        label_var = label_var.repeat(self.num_copies)
        model = train_model
        step_alpha = self.step_alpha
        g = 0.0
        mu = 1.0
        steps = self.num_steps
        step = 0
        eps =self.eps
        while step < steps:
            input_var.grad = None
            #train adversarial example
            output_f1 = model(self.SIA_transform(input_var))
            output_f2 = model(self.T_net(self.SIA_transform(input_var)))
            # output_f1 = model(input_var)
            # output_f2 = model(self.T_net(input_var))   
            loss_f1 = self.loss_fn(output_f1, label_var)
            loss_f2 = self.loss_fn(output_f2, label_var)
            loss_fool = loss_f1 + loss_f2
            loss_fool.backward()

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
