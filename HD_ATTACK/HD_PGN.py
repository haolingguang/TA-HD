import torch
import torch.nn as nn


class HD_PGN:
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
        
    def train_T(self, train_model, input, true_label, writer, loss_mode, T_Net_name, global_step):
        '''Train HD'''
        input_var = input.clone().detach()
        label_var = true_label.clone()
        input_clear = input.clone().detach()

        model = train_model
        step_alpha = self.step_alpha
        mu = 1.0
        steps = self.num_steps
        eps =self.eps
        N = 20
        zeta = 3.0
        delta = 0.5
        
        grad = torch.zeros_like(input_var).detach().cuda()
        for step in range(steps):
            avg_grad = torch.zeros_like(input_var).detach().cuda()
            for _ in range(N):
                x_near = input_var + torch.rand_like(input_var).uniform_(-eps * zeta, eps * zeta)
                x_near = torch.autograd.Variable(x_near, requires_grad = True)
                output_x_near1 = model(x_near)
                output_x_near2 = model(self.T_net(x_near))
                loss_x_near1 = self.loss_fn(output_x_near1, label_var)
                loss_x_near2 = self.loss_fn(output_x_near2, label_var)
                loss_x_near = loss_x_near1 + loss_x_near2
                g1 = torch.autograd.grad(loss_x_near, x_near,
                                            retain_graph=False, create_graph=False)[0]
                x_star = x_near.detach() + step_alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)

                nes_x = x_star.detach()
                nes_x = torch.autograd.Variable(nes_x, requires_grad = True)
                output_nes_x1 = model(nes_x)
                output_nes_x2 = model(self.T_net(nes_x))
                loss_nes_x1 = self.loss_fn(output_nes_x1, label_var)
                loss_nes_x2 = self.loss_fn(output_nes_x2, label_var)
                loss_nes_x = loss_nes_x1 + loss_nes_x2
                g2 = torch.autograd.grad(loss_nes_x, nes_x,
                                            retain_graph=False, create_graph=False)[0]

                avg_grad += 1/N * ((1-delta)*g1 + delta*g2)
            noise = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
            grad = mu * grad + noise
            # grad = noise
            
            input_var = input_var + step_alpha * torch.sign(grad)
            input_var = torch.clamp(input_var, -1.0, 1.0)
        

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
        
        input_var = input.clone().detach()
        label_var = true_label.clone()
        model = train_model
        step_alpha = self.step_alpha
        g = 0.0
        mu = 1.0
        steps = self.num_steps
        step = 0
        eps =self.eps
        N = 20
        zeta = 3.0
        delta = 0.5
        
        grad = torch.zeros_like(input_var).detach().cuda()
        for step in range(steps):
            avg_grad = torch.zeros_like(input_var).detach().cuda()
            for _ in range(N):
                x_near = input_var + torch.rand_like(input_var).uniform_(-eps * zeta, eps * zeta)
                x_near = torch.autograd.Variable(x_near, requires_grad = True)
                output_x_near1 = model(x_near)
                output_x_near2 = model(self.T_net(x_near))
                loss_x_near1 = self.loss_fn(output_x_near1, label_var)
                loss_x_near2 = self.loss_fn(output_x_near2, label_var)
                loss_x_near = loss_x_near1 + loss_x_near2
                g1 = torch.autograd.grad(loss_x_near, x_near, retain_graph=False, create_graph=False)[0]
                x_star = x_near.detach() + step_alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)

                nes_x = x_star.detach()
                nes_x = torch.autograd.Variable(nes_x, requires_grad = True)
                output_nes_x1 = model(nes_x)
                output_nes_x2 = model(self.T_net(nes_x))
                loss_nes_x1 = self.loss_fn(output_nes_x1, label_var)
                loss_nes_x2 = self.loss_fn(output_nes_x2, label_var)
                loss_nes_x = loss_nes_x1 + loss_nes_x2
                g2 = torch.autograd.grad(loss_nes_x, nes_x, retain_graph=False, create_graph=False)[0]
                
                avg_grad += 1/N * ((1-delta)*g1 + delta*g2)
            noise = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
            grad = mu * grad + noise
            
            input_var = input_var + step_alpha * torch.sign(grad)
            input_var = torch.clamp(input_var, -1.0, 1.0)
            
        return input_var.detach()
    
