import torch
from torch import autograd


class FGSM_GRA:
    def __init__(self, max_epsilon=16, norm=float('inf'), num_steps=None):

        self.eps = 2.0 * max_epsilon / 255.0
        self.num_steps = num_steps
        self.norm = norm
        self.step_alpha = self.eps / self.num_steps
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.cuda()
    
    
    def batch_grad(self, x_adv, y, model, sample_num=20, beta_eps=3.5 * 16/255, grad=None):
        for _ in range(sample_num):
            # 在x_adv附近进行采样
            x_near = x_adv.clone().detach() + torch.rand_like(x_adv).uniform_(-beta_eps, beta_eps)
            x_near = autograd.Variable(x_near, requires_grad = True)
            output = model(x_near)
            loss = self.loss_fn(output, y)
            sample_g = autograd.grad(loss, x_near,retain_graph=False, create_graph=False)[0]
            grad += sample_g
        return grad
            

    def non_target_attack(self, train_model, input, true_label, batch_idx=0, steps=0):

        input_var = autograd.Variable(input,requires_grad = True)
        label_var = true_label.clone()
        model = train_model
        step_alpha = self.step_alpha
        # g = 0.0
        mu = 1.0
        steps = self.num_steps
        step = 0
        eps =self.eps
        sample_num = 20
        
        # 用于记录第t和t+1次的梯度
        grad_t = torch.zeros_like(input)
        grad_t_plus = torch.zeros_like(input)
        
        # 用于记录学习率的衰减因子
        m = torch.ones_like(input) * 1/0.94
        
        while step < steps:
            # input_var.grad = None
            output = model(input_var)
            loss = self.loss_fn(output, label_var)
            
            # 计算对抗样本的梯度
            adv_grad = autograd.grad(loss, input_var,retain_graph=False, create_graph=False)[0]
            
            # 计算在对抗样本附近采样的样本的平均梯度
            global_grad = self.batch_grad(input_var.detach().clone(), label_var.clone(), model, sample_num, 3.5*eps, grad=torch.zeros_like(adv_grad))
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
            step += 1
        return input_var.detach()    
     