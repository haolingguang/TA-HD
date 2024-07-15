import torch
import torch.nn as nn


class HD_GRA:
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
    
    
    def batch_grad(self, x_adv, y, model, sample_num=20, beta_eps=3.5 * 16/255, grad=None):
        for _ in range(sample_num):
            # 在x_adv附近进行采样
            x_near = x_adv.clone().detach() + torch.rand_like(x_adv).uniform_(-beta_eps, beta_eps)
            x_near = torch.autograd.Variable(x_near, requires_grad = True)
            output_x_near1 = model(x_near)
            output_x_near2 = model(self.T_net(x_near))
            loss_x_near1 = self.loss_fn(output_x_near1, y)
            loss_x_near2 = self.loss_fn(output_x_near2, y)
            loss_x_near = loss_x_near1 + loss_x_near2
            sample_g = torch.autograd.grad(loss_x_near, x_near,retain_graph=False, create_graph=False)[0]
            grad += sample_g
        return grad
    
        
    def train_T(self, train_model, input, true_label, writer, loss_mode, T_Net_name, global_step):
        '''Train HD'''
        
        input_var = torch.autograd.Variable(input, requires_grad=True)
        label_var = torch.autograd.Variable(true_label)
        input_clear = input.clone().detach()

        model = train_model
        step_alpha = self.step_alpha
        mu = 1.0
        steps = self.num_steps
        eps =self.eps
        
        sample_num = 20
        
        # 用于记录第t和t+1次的梯度
        grad_t = torch.zeros_like(input)
        grad_t_plus = torch.zeros_like(input)
        
        # 用于记录学习率的衰减因子
        m = torch.ones_like(input) * 1/0.94
        
        for step in range(steps):
            # input_var.grad = None
            output_adv1 = model(input_var)
            output_adv2 = model(self.T_net(input_var))
            
            loss_adv1 = self.loss_fn(output_adv1, label_var)
            loss_adv2 = self.loss_fn(output_adv2, label_var)
            loss_adv = loss_adv1 + loss_adv2
                        
            # 计算对抗样本的梯度
            adv_grad = torch.autograd.grad(loss_adv, input_var,retain_graph=False, create_graph=False)[0]
            
            # 计算在对抗样本附近采样的样本的平均梯度
            global_grad = self.batch_grad(input_var, label_var, model, sample_num, 3.5*eps, grad=torch.zeros_like(adv_grad))
            sam_grad  = global_grad/sample_num
            
            # 计算余弦相似度, 并更新当前梯度
            cossim = torch.sum(adv_grad * sam_grad) / (torch.sqrt(torch.sum(adv_grad ** 2)) * torch.sqrt(torch.sum(sam_grad ** 2)))
            current_grad = cossim * adv_grad + (1-cossim) * sam_grad 
            
            # 根据第t次迭代的梯度和当前梯度计算t+1次的梯度，并计算调整学习率的衰减因子
            grad_t_plus = mu * grad_t + current_grad / torch.abs(current_grad).mean([1, 2, 3], keepdim=True)
            eqm = torch.eq(torch.sign(grad_t),torch.sign(grad_t_plus)).float()
            dim = torch.ones_like(eqm) - eqm
            m = m * (eqm + dim * 0.94)
            m = torch.clamp(m, 0.0, 1.0)
            # 更新对抗样本
            input_var = input_var + step_alpha * m * torch.sign(grad_t_plus)
            input_var = torch.clamp(input_var, -1.0, 1.0)
            # 更新梯度记录
            grad_t = grad_t_plus
        
        self.T_optimizer.zero_grad()
        output_T1 = model(self.T_net(input_var))
        output_T2 = model(self.T_net(input_clear))
        loss_T1 = self.loss_fn(output_T1, label_var)
        loss_T2 = self.loss_fn(output_T2, label_var)
        
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
        model = train_model
        step_alpha = self.step_alpha
        mu = 1.0
        steps = self.num_steps
        eps =self.eps
        
        sample_num = 20
        
        # 用于记录第t和t+1次的梯度
        grad_t = torch.zeros_like(input)
        grad_t_plus = torch.zeros_like(input)
        
        # 用于记录学习率的衰减因子
        m = torch.ones_like(input) * 1/0.94
        
        for step in range(steps):
            # input_var.grad = None
            output_adv1 = model(input_var)
            output_adv2 = model(self.T_net(input_var))
            
            loss_adv1 = self.loss_fn(output_adv1, label_var)
            loss_adv2 = self.loss_fn(output_adv2, label_var)
            loss_adv = loss_adv1 + loss_adv2
                        
            # 计算对抗样本的梯度
            adv_grad = torch.autograd.grad(loss_adv, input_var,retain_graph=False, create_graph=False)[0]
            
            # 计算在对抗样本附近采样的样本的平均梯度
            global_grad = self.batch_grad(input_var, label_var, model, sample_num, 3.5*eps, grad=torch.zeros_like(adv_grad))
            sam_grad  = global_grad/sample_num
            
            # 计算余弦相似度, 并更新当前梯度
            cossim = torch.sum(adv_grad * sam_grad) / (torch.sqrt(torch.sum(adv_grad ** 2)) * torch.sqrt(torch.sum(sam_grad ** 2)))
            current_grad = cossim * adv_grad + (1-cossim) * sam_grad 
            
            # 根据第t次迭代的梯度和当前梯度计算t+1次的梯度，并计算调整学习率的衰减因子
            grad_t_plus = mu * grad_t + current_grad / torch.abs(current_grad).mean([1, 2, 3], keepdim=True)
            eqm = torch.eq(torch.sign(grad_t),torch.sign(grad_t_plus)).float()
            dim = torch.ones_like(eqm) - eqm
            m = m * (eqm + dim * 0.94)
            m = torch.clamp(m, 0.0, 1.0)
            # 更新对抗样本
            input_var = input_var + step_alpha * m * torch.sign(grad_t_plus)
            input_var = torch.clamp(input_var, -1.0, 1.0)
            # 更新梯度记录
            grad_t = grad_t_plus
            
        return input_var.detach()
    
