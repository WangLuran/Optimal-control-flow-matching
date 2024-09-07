import gc
import io
import os
import time

import numpy as np
import logging
import lpips

# Keep the import below for registering all model definitions
from RectifiedFlow.models import ddpm, ncsnv2, ncsnpp
from RectifiedFlow.models import utils as mutils
from RectifiedFlow.models.ema import ExponentialMovingAverage
from absl import flags
import torch
from torchvision.utils import make_grid, save_image
from RectifiedFlow.utils import save_checkpoint, restore_checkpoint
import RectifiedFlow.datasets as datasets

from RectifiedFlow.models.utils import get_model_fn
from RectifiedFlow.models import utils as mutils

from .flowgrad_utils import get_img, embed_to_latent, clip_semantic_loss, save_img, generate_traj, flowgrad_optimization

FLAGS = flags.FLAGS

@torch.no_grad()
def generate_traj_oc(dynamic, z0, u, N):
  traj = []

  # Initial sample
  z = z0.detach().clone()
  traj.append(z.detach().clone().cpu())
  batchsize = z0.shape[0]

  dt = 1./N
  eps = 1e-3
  pred_list = []
  for i in range(N):
    z += u/N
    t = torch.ones(z0.shape[0], device=z0.device) * i / N * (1.-eps) + eps
    pred = dynamic(z, t*999)
    #print('compare',torch.sum(dynamic(z, t*(N-1))),torch.sum(u))
    z = z.detach().clone() + pred * dt
      
    traj.append(z.detach().clone())

    pred_list.append(pred.detach().clone().cpu())

    return traj

def flowgrad_optimization_oc(z0, dynamic, generate_traj, N, L_N,  number_of_iterations, alpha,
                                  beta):
    device = z0.device
    shape = z0.shape
    traj_org = generate_traj(dynamic, z0=z0, N=N)

    X1 = traj_org[-1]

    #optimal control
    lambda_ = torch.randn_like(X1).copy_(X1) 
    lambda_.requires_grad_(True)
    if not lambda_.requires_grad:
        print(f"lambda is not properly set")

    for i in range(number_of_iterations):
        input_tensor = lambda_ + X1

        # R_output = -L_N(input_tensor) #+ 1e-4*grad_F*lambda_
        R_output = torch.square(L_N(input_tensor))

        score_sum = R_output.sum()
    
        # Compute the gradient of R(lambda + X) with respect to X
        grad_R = torch.autograd.grad(outputs=score_sum, inputs=lambda_, create_graph=False)[0]

        lambda_ = beta* lambda_ + alpha * grad_R

        if i%100 == 0:
            print(i, L_N(input_tensor), torch.norm(grad_R))

    out = X1 + lambda_

    # print('grad_F',grad_F)

    return lambda_, out

def flowgrad_optimization_oc_d(z0, u_ind, dynamic, generate_traj, N, L_N,  number_of_iterations, alpha,
                                  beta):
    device = z0.device
    shape = z0.shape
    u = {}
    eps = 1e-3 # default: 1e-3
    for ind in u_ind:
        u[ind] = torch.zeros_like(z0).to(z0.device)
        u[ind].requires_grad = True
        u[ind].grad = torch.zeros_like(u[ind], device=u[ind].device)
    for i in range(number_of_iterations):
        ### get the forward simulation result and the non-uniform discretization trajectory
        ### non_uniform_set: indices and interval length (t_{j+1} - t_j)
        z_traj, non_uniform_set = generate_traj(dynamic, z0, u=u, N=N, straightness_threshold=0)
        # print(non_uniform_set)

        t_s = time.time()
        ### use lambda to store \nabla L
        inputs = torch.zeros(z_traj[-1].shape, device=device)
        inputs.data = z_traj[-1].to(device).detach().clone()
        inputs.requires_grad = True
        loss = -L_N(inputs)
        lam = torch.autograd.grad(loss, inputs)[0]
        lam = lam.detach().clone()

        print('iteration:', i)
        # print('   inputs:', inputs.view(-1).detach().cpu().numpy())
        print('   L:%.6f'%loss.detach().cpu().numpy())
        # print('   lambda:', lam.reshape(-1).detach().cpu().numpy())
        
        eps = 1e-3 # default: 1e-3
        g_old = None
        d = []
        for j in range(N-1, -1, -1):
            # if j in non_uniform_set['indices']:
            #   assert j in u_ind
            # else:
            #   continue

            ### compute lambda: correct vjp version
            inputs = torch.zeros(lam.shape, device=device)
            inputs.data = z_traj[j].to(device).detach().clone()
            inputs.requires_grad = True
            t = (torch.ones((1, )) * j / N * (1.-eps) + eps) * 999
            func = lambda x: (x.contiguous().reshape(shape) + u[j].detach().clone() + \
                              dynamic(x.contiguous().reshape(shape) + u[j].detach().clone(), t.detach().clone()) * non_uniform_set['length'][j] / N).view(-1)
            output, vjp = torch.autograd.functional.vjp(func, inputs=inputs.view(-1), v=lam.detach().clone().reshape(-1))
            lam = vjp.detach().clone().contiguous().reshape(shape)
            
            u[j].grad = lam.detach().clone()
            del inputs
            if j == 0: break
        
        print('BP time:', time.time() - t_s)

        for ind in u.keys():
          u[ind] = u[ind]*beta + alpha*u[ind].grad

    opt_u = {}
    for ind in u.keys():
        opt_u[ind] = u[ind].detach().clone()

    return opt_u

def flowgrad_edit(config, text_prompt, alpha, model_path, data_loader, output_folder="output"):
  clip_scores = []
  lpips_scores = []
  for batch in data_loader:
    images, labels = batch
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    state = restore_checkpoint(model_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    model_fn = mutils.get_model_fn(score_model, train=False)

    # Load the image to edit
    original_img = get_img(image_path)  
  
    # log_folder = os.path.join(output_folder, 'figs')
    # print('Images will be saved to:', log_folder)
    # if not os.path.exists(log_folder): os.makedirs(log_folder)
    # save_img(original_img, path=os.path.join(log_folder, 'original.png'))

    # Get latent code of the image and save reconstruction
    original_img = original_img.to(config.device)
    clip_loss = clip_semantic_loss(text_prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)  
    clip_loss_1 = clip_semantic_loss(text_prompt, original_img, config.device, alpha=1., inverse_scaler=inverse_scaler)  

    lpips_f = lpips.LPIPS(net='alex').to(config.device) # or 'vgg', 'squeeze'

    t_s = time.time()
    latent = embed_to_latent(model_fn, scaler(original_img))
    traj = generate_traj(model_fn, latent, N=100)
  #   save_img(inverse_scaler(traj[-1]), path=os.path.join(log_folder, 'reconstruct.png'))
  #   print('Finished getting latent code and reconstruction; image saved.')
  
    # Edit according to text prompt
    u_ind = [i for i in range(100)]
    u_opt = flowgrad_optimization_oc_d(latent, u_ind, model_fn, generate_traj, N=100, L_N=clip_loss.L_N,  number_of_iterations=10, alpha=10,
                                  beta=0.985) 
    # opt_u = flowgrad_optimization(latent, u_ind, model_fn, generate_traj, N=100, L_N=clip_loss.L_N, u_init=None,  number_of_iterations=10, straightness_threshold=5e-3, lr=10.0) 
    # traj_gd = generate_traj(model_fn, z0=latent, u=opt_u, N=100)

    traj_oc = generate_traj(model_fn, z0=latent, u=u_opt, N=100)

    clip_scores.append(clip_loss_1.L_N(traj_oc[-1]).sum())
    lpips_scores.append(lpips_f(traj_oc[-1], traj[-1]).sum())

    break


  return sum(clip_scores)/len(clip_scores), sum(lpips_scores)/len(lpips_scores)

#   print('OC results:')
# #   print(clip_loss_1.L_N(p_generated))
#   print(clip_loss_1.L_N(traj_oc[-1]))
#   print(lpips_f(traj_oc[-1], traj[-1]))
#   print('GD results:')
#   print(clip_loss_1.L_N(traj_gd[-1]))
#   print(lpips_f(traj_gd[-1], traj[-1]))
# #   print('Original:')
# #   print(clip_loss_1.L_N(traj[-1]))
# #   print(lpips_f(traj[-1], traj[-1]))
#   print('Total time:', time.time() - t_s)
#   save_img(inverse_scaler(traj_oc[-1]), path=os.path.join(log_folder, 'optimized_oc_d.png'))
#   print('Finished Editting; images saved.')
#   # save_img(inverse_scaler(lambda_), path=os.path.join(log_folder, 'optimized_feature.png'))
#   # print('Finished Editting; images saved.')
#   save_img(inverse_scaler(traj_gd[-1]), path=os.path.join(log_folder, 'optimized_gd.png'))
#   print('Finished Editting; images saved.')