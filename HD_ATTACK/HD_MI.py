import torch
import torch.nn as nn




class HD_MI:
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
        while step < steps:
            input_var.grad = None
            self.T_optimizer.zero_grad()
            #train adversarial example
            output_f1 = model(self.T_net(input_var))
            output_f2 = model(input_var)  
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
        loss_T1 = self.loss_fn(output_T1, label_var)
        loss_T2 = self.loss_fn(output_T2, label_var)
        
        x = self.T_net(input_var)
        loss_T3 = input_clear-x
        
        # if T_Net_name == "Conv_Net":
        #     loss_T3 = input_var-x   # 这里是原来的ATTA论文中吴卫兵用的loss，期望这个T_Net不要矫正过剩
        # else:
        #     loss_T3 = input_clear-x  # 这个默认是我们提出的方法中的loss，即降噪后的样本和清洁样本靠近
        
        loss_T3 = torch.mean(torch.norm(torch.flatten(loss_T3, start_dim=1), p=2 ,dim=1))
        loss_T = loss_T1 + loss_T2 + loss_T3
        
        # loss_T4 = self.loss_T(output_T1,output_T2)  # #这个是CGD loss
        # MSE_loss = self.loss_T(x,input_var)
        # loss_T4 = metrics.mutual_info_score(self.T_net(input_var), input_clear)
        
        # if loss_mode=='CGD':
        #     loss_T = loss_T1 + loss_T2
        # elif loss_mode=='LGD_CGD':
        #     loss_T = loss_T1 + loss_T2 + loss_T3
        
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
        self.T_optimizer.zero_grad()
        return loss_T1.item(), loss_T2.item(), loss_T3.item()

        # return input_adv
    def test_T(self, train_model, input, true_label):
        '''Test the performance of HD'''
        
        input_var = torch.autograd.Variable(input, requires_grad=True)
        label_var = torch.autograd.Variable(true_label)
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
            output_f1 = model(input_var)
            output_f2 = model(self.T_net(input_var))   
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
    
