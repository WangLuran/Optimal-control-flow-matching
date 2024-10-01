import gc
import io
import os
import time

import numpy as np
import logging
import lpips
import json
from tqdm import tqdm
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
# from id_loss.loss_fn import IDLoss

import warnings
warnings.filterwarnings("ignore")
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0'



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

def dflow_optimization(z0, dynamic, N, L_N,  number_of_iterations, alpha):
    device = z0.device
    shape = z0.shape
    batch_size = z0.shape[0]
    # z0.requires_grad = True

    dt = 1./N
    eps = 1e-3 # default: 1e-3

    L_best = 0

    def grad_calculate(z0):
      z_traj, non_uniform_set = generate_traj(dynamic, z0, N=N, straightness_threshold=0)

      t_s = time.time()
      inputs = torch.zeros(z_traj[-1].shape, device=device)
      inputs.data = z_traj[-1].to(device).detach().clone()
      inputs.requires_grad = True
      loss = -L_N(inputs)
      lam = torch.autograd.grad(loss, inputs)[0]
      lam = lam.detach().clone()
        
      eps = 1e-3 # default: 1e-3
      g_old = None
      d = []
      for j in range(N-1, -1, -1):

        inputs = torch.zeros(lam.shape, device=device)
        inputs.data = z_traj[j].to(device).detach().clone()
        inputs.requires_grad = True
        t = (torch.ones((batch_size, )) * j / N * (1.-eps) + eps) * 999
        func = lambda x: (x.contiguous().reshape(shape) + \
                              dynamic(x.contiguous().reshape(shape), t.detach().clone()) * non_uniform_set['length'][j] / N).view(-1)
        output, vjp = torch.autograd.functional.vjp(func, inputs=inputs.view(-1), v=lam.detach().clone().reshape(-1))
        lam = vjp.detach().clone().contiguous().reshape(shape)
            
        del inputs
        if j == 0: break
        
      return lam

    L_best = 0

    # optimizer = torch.optim.LBFGS([z0], lr=alpha, max_iter=number_of_iterations, history_size=10, line_search_fn='strong_wolfe')
    # optimizer.step(closure)

    for i in range(number_of_iterations):
      z_traj, _ = generate_traj(dynamic, z0, N=N, straightness_threshold=0)
      loss = -L_N(z_traj[-1])

      if loss.detach().cpu().numpy() > L_best:
          z_best = z0
          L_best = loss.detach().cpu().numpy()

      z0 = z0 + alpha*grad_calculate(z0)
      print(f'Iter {i}: Loss {loss.item():.4f}')

    return z_best

def dflow_optimization_lbfgs(z0, dynamic, N, L_N, max_iter, max_step=5, lr=1, verbose=False):
    device = z0.device
    shape = z0.shape
    batch_size = z0.shape[0]
    # z0.requires_grad = True

    cnt = 0
    dt = 1./N
    eps = 1e-3 # default: 1e-3

    # def loss_fn(cur_r0):
    #     r = cur_r0
    #     dt = 1./N
    #     eps = 1e-3
    #     for i in range(N):
    #       t = torch.ones(z0.shape[0], device=z0.device) * i / N * (1.-eps) + eps
    #       pred = dynamic(r, t*999)
    #       r = r + pred * dt

    #     return r.detach(), L_N(r)

    def loss_fn(cur_r0):
        r = cur_r0
        # ts = torch.linspace(*tlist, n_step + 1, device=r.device) * (1. - eps) + eps
        # print('ts', ts)
        dt = 1 / N
        # print('dt', dt)
        for i in range(N):
          t = torch.ones(z0.shape[0], device=z0.device) * i / N * (1.-eps) + eps
          # print(i, t, t * 999)
          # TODO: check if t is correct
          r = r + dynamic(r, t * 999) * dt

        return r.detach(), L_N(r)

    # def closure():
    #     nonlocal cnt, r0_opt
    #     cnt += 1
    #     optimizer.zero_grad()
    #     r0_opt.requires_grad_(False)
    #     _, loss = loss_fn(r0_opt)
    #     loss.backward()
    #     if verbose:
    #         print(f'Iter {cnt}: Loss {loss.item():.4f}')
    #     return loss
    
    def closure():
      nonlocal cnt
      cnt += 1
      optimizer.zero_grad()
      _, loss = loss_fn(r0_opt)
      if torch.isnan(loss).any():
          return torch.tensor(1e5, device=r0_opt.device)
      loss.backward(retain_graph=True)
      # clip_grad_norm_(r0_opt, max_grad_norm)
      if verbose:
          print(f'Iter {cnt}: Loss {loss.item():.4f}')
      return loss

    r0_opt = z0.detach().clone()
    r0_opt.requires_grad_(True)
    optimizer = torch.optim.LBFGS([r0_opt], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')
    
    for step in range(max_step):
      loss = optimizer.step(closure)
      print(f'Step {step}: Loss {loss.item():.4f}')

    r0_opt = r0_opt.detach()
    with torch.no_grad():
      r1_opt, _ = loss_fn(r0_opt)

    return r1_opt, r0_opt

def flowgrad_optimization_oc_d(z0, u_ind, dynamic, generate_traj, N, L_N,  number_of_iterations, alpha,
                                  beta):
    device = z0.device
    shape = z0.shape
    batch_size = shape[0]
    # print('batch_size',batch_size)
    u = {}
    eps = 1e-3 # default: 1e-3
    for ind in u_ind:
        u[ind] = torch.zeros_like(z0).to(z0.device)
        u[ind].requires_grad = True
        u[ind].grad = torch.zeros_like(u[ind], device=u[ind].device)

    L_best = 0
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

        if loss.detach().cpu().numpy() > L_best:
          opt_u = {}
          for ind in u.keys():
              opt_u[ind] = u[ind].detach().clone()
          L_best = loss.detach().cpu().numpy()

        # print('iteration:', i)
        # # print('   inputs:', inputs.view(-1).detach().cpu().numpy())
        # print('   L:%.6f'%loss.detach().cpu().numpy())
        # # print('   lambda:', lam.reshape(-1).detach().cpu().numpy())
        
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
            t = (torch.ones((batch_size, )) * j / N * (1.-eps) + eps) * 999
            func = lambda x: (x.contiguous().reshape(shape) + u[j].detach().clone() + \
                              dynamic(x.contiguous().reshape(shape) + u[j].detach().clone(), t.detach().clone()) * non_uniform_set['length'][j] / N).view(-1)
            output, vjp = torch.autograd.functional.vjp(func, inputs=inputs.view(-1), v=lam.detach().clone().reshape(-1))
            lam = vjp.detach().clone().contiguous().reshape(shape)
            
            u[j].grad = lam.detach().clone()
            del inputs
            if j == 0: break
        
        # print('BP time:', time.time() - t_s)

        for ind in u.keys():
          u[ind] = u[ind]*beta + batch_size*alpha*u[ind].grad

    return opt_u

def dflow_edit(config, text_prompts, alpha, model_path, data_loader):
  clip_scores = []
  lpips_scores = []
  id_scores = []
  clip_scores_gd = []
  lpips_scores_gd = []
  id_scores_gd = []
  for batch in data_loader:
    images = batch[:,0,:,:,:]
    batch_size = images.shape[0]
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
    # img = get_img('demo/celeba.jpg')
    # images = img
    # print('o_img shape',img.shape)
    original_img = images  
  
    log_folder = os.path.join('output', 'figs')
    print('Images will be saved to:', log_folder)
    # if not os.path.exists(log_folder): os.makedirs(log_folder)
    save_img(original_img, path=os.path.join(log_folder, 'original.png'))

    # Get latent code of the image and save reconstruction
    for text_prompt in text_prompts:
      original_img = original_img.to(config.device)
      clip_loss = clip_semantic_loss(text_prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)  
      clip_loss_1 = clip_semantic_loss(text_prompt, original_img, config.device, alpha=1., inverse_scaler=inverse_scaler)  
      # id_loss = IDLoss(device=config.device)

      lpips_f = lpips.LPIPS(net='alex').to(config.device) # or 'vgg', 'squeeze'

      t_s = time.time()
      latent = embed_to_latent(model_fn, scaler(original_img))
      traj = generate_traj(model_fn, latent, N=100)
  
      # Edit according to text prompt
      print('optimization starts')
      z0_d = dflow_optimization_lbfgs(latent, model_fn, N=100, L_N=clip_loss_1.L_N,  max_iter=5, lr=1)

      traj_oc = generate_traj(model_fn, z0=z0_d, N=100)

      print('dif', (z0_d-latent).sum())

      save_img(inverse_scaler(traj_oc[-1]), path=os.path.join(log_folder, 'optimized_dflow.png'))
    
      clip_scores.append(clip_loss_1.L_N(traj_oc[-1]).detach().cpu().numpy().sum())
      lpips_scores.append(lpips_f(traj_oc[-1], traj[-1]).detach().cpu().numpy().mean())
      # id_scores.append(1. - id_loss(traj[-1], traj_oc[-1]).detach().cpu().numpy().mean())

      print('text prompt', text_prompt)

      print('total_clip_loss',sum(clip_scores)/len(clip_scores))
      print('total_lpips_f',sum(lpips_scores)/len(lpips_scores))
      print('total_id',sum(id_scores)/len(id_scores))
      print('num', len(clip_scores)/5)

  return sum(clip_scores)/len(clip_scores), sum(lpips_scores)/len(lpips_scores), sum(id_scores)/len(id_scores)#,sum(clip_scores_gd)/len(clip_scores_gd), sum(lpips_scores_gd)/len(lpips_scores_gd),sum(id_scores_gd)/len(id_scores_gd)


# define a context manager which can count time cost
from contextlib import contextmanager

@contextmanager
def timer(name):
   print('running', name, '...')
   start_time = time.time()
   yield
   elapsed_time = time.time() - start_time
   print(f'\t{name} costs: {elapsed_time:.4f} s')


def dflow_edit_single(config, text_prompt, alpha, model_path, image_path, output_folder='output'):
  image = get_img(image_path)  
  batch_size = 1

  ts = time.time()
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

  log_folder = os.path.join(output_folder, 'figs')
  print('Images will be saved to:', log_folder)
  if not os.path.exists(log_folder): os.makedirs(log_folder)
  save_img(image, path=os.path.join(log_folder, 'original.png'))

  original_img = image.to(config.device)
  with timer('clip loss'):
    clip_loss = clip_semantic_loss(text_prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)  
  with timer('clip loss 1'):
    clip_loss_1 = clip_semantic_loss(text_prompt, original_img, config.device, alpha=1., inverse_scaler=inverse_scaler)  
  with timer('lpips'):
    lpips_f = lpips.LPIPS(net='alex').to(config.device) # or 'vgg', 'squeeze'

  # TODO: this step is very slow, consider preprocessing all images
  with timer('embed'):
    latent = embed_to_latent(model_fn, scaler(original_img))
  # torch.save(latent, 'latent.pt')
  save_img(inverse_scaler(latent), path=os.path.join(log_folder, 'latent.png'))
  # latent = get_img(os.path.join(log_folder, 'latent.png')).to(config.device)

  N = 5
  with timer('generate traj'):
    traj = generate_traj(model_fn, latent, N=N)
  recover_image = inverse_scaler(traj[-1])
  save_img(recover_image, path=os.path.join(log_folder, f'recover_{N}.png'))
  # torch.save(traj, 'traj.pt')
  # traj = torch.load('traj.pt')
  # for x in traj:
  #    x.to(config.device)

  lr = 1
  lbfgs_max_iter = 20
  opt_max_step = 5
  # Edit according to text prompt
  with timer('optimization'):
    z1_d, z0_d = dflow_optimization_lbfgs(latent, model_fn, N=N, L_N=clip_loss_1.L_N,  max_iter=lbfgs_max_iter, max_step=opt_max_step, lr=lr)
  save_img(inverse_scaler(z0_d), path=os.path.join(log_folder, f'z0_d_{N}.png'))

  # with timer('generate traj with z0_d'):
  #   traj_oc = generate_traj(model_fn, z0=z0_d, N=N)

  print('dif', (z0_d-latent).sum())

  save_img(inverse_scaler(z1_d), path=os.path.join(log_folder, 'optimized_dflow.png'))

  clip_loss = clip_loss_1.L_N(z1_d).detach().cpu().numpy()
  lpips_score = lpips_f(z1_d, traj[-1]).detach().cpu().numpy()

  print('text prompt', text_prompt)

  print('clip loss', clip_loss)
  print('lpips score', lpips_score)
  print('total time', time.time() - ts)


def flowgrad_edit(config, text_prompts, alpha, model_path, data_loader):
  clip_scores = []
  lpips_scores = []
  id_scores = []
  clip_scores_gd = []
  lpips_scores_gd = []
  id_scores_gd = []
  for batch in data_loader:
    images = batch[:,0,:,:,:]
    batch_size = images.shape[0]
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
    # img = get_img('demo/celeba.jpg')
    # images = img
    # print('o_img shape',img.shape)
    original_img = images  
  
    log_folder = os.path.join('output', 'figs')
    print('Images will be saved to:', log_folder)
    # if not os.path.exists(log_folder): os.makedirs(log_folder)
    # save_img(original_img, path=os.path.join(log_folder, 'original.png'))

    # Get latent code of the image and save reconstruction
    for text_prompt in text_prompts:
      original_img = original_img.to(config.device)
      clip_loss = clip_semantic_loss(text_prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)  
      clip_loss_1 = clip_semantic_loss(text_prompt, original_img, config.device, alpha=1., inverse_scaler=inverse_scaler)  
      # id_loss = IDLoss(device=config.device)

      lpips_f = lpips.LPIPS(net='alex').to(config.device) # or 'vgg', 'squeeze'

      t_s = time.time()
      latent = embed_to_latent(model_fn, scaler(original_img))
      traj = generate_traj(model_fn, latent, N=100)
      save_img(inverse_scaler(traj[-1]), path=os.path.join(log_folder, 'original.png'))
  
      # Edit according to text prompt
      print('optimization starts')
      u_ind = [i for i in range(100)]
      u_opt = flowgrad_optimization_oc_d(latent, u_ind, model_fn, generate_traj, N=100, L_N=clip_loss.L_N,  number_of_iterations=15, alpha=2.5,#first 3, second 2.75, third 2.5
                                  beta=0.995) #first is 0.990, second is 0.9995, third is 0.995; first is 0.9925 third 0.995 last is 0.990
      # opt_u = flowgrad_optimization(latent, u_ind, model_fn, generate_traj, N=100, L_N=clip_loss.L_N, u_init=None,  number_of_iterations=10, straightness_threshold=5e-3, lr=10.0) 
      # traj_gd = generate_traj(model_fn, z0=latent, u=opt_u, N=100)

      traj_oc = generate_traj(model_fn, z0=latent, u=u_opt, N=100)

      save_img(inverse_scaler(traj_oc[-1]), path=os.path.join(log_folder, 'optimized_oc_d_t.png'))
    
      clip_scores.append(clip_loss_1.L_N(traj_oc[-1]).detach().cpu().numpy().sum())
      lpips_scores.append(lpips_f(traj_oc[-1], traj[-1]).detach().cpu().numpy().mean())
      # id_scores.append(1. - id_loss(traj[-1], traj_oc[-1]).detach().cpu().numpy().mean())

    print('total_clip_loss',sum(clip_scores)/len(clip_scores))
    print('total_lpips_f',sum(lpips_scores)/len(lpips_scores))
    print('total_id',sum(id_scores)/len(id_scores))
    print('num', len(clip_scores)/5)

  return sum(clip_scores)/len(clip_scores), sum(lpips_scores)/len(lpips_scores), sum(id_scores)/len(id_scores)#,sum(clip_scores_gd)/len(clip_scores_gd), sum(lpips_scores_gd)/len(lpips_scores_gd),sum(id_scores_gd)/len(id_scores_gd)


def flowgrad_edit_single(config, text_prompt, alpha, model_path, image_path, opt_img_path=None):   
  
  
  
  # log_folder = os.path.join(output_folder, 'figs')
  # print('Images will be saved to:', log_folder)
  # if not os.path.exists(log_folder): os.makedirs(log_folder)
  # save_img(image, path=os.path.join(log_folder, 'original.png'))

  
  



  
  # save_img(inverse_scaler(traj[-1]), path=os.path.join(log_folder, 'recover.png'))
  
  
  # opt_u = flowgrad_optimization(latent, u_ind, model_fn, generate_traj, N=100, L_N=clip_loss.L_N, u_init=None,  number_of_iterations=10, straightness_threshold=5e-3, lr=10.0) 
  # traj_gd = generate_traj(model_fn, z0=latent, u=opt_u, N=100)


  


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
    # file_path = 'output/results.json'

    # with open(file_path, 'r') as json_file:
    #   data = json.load(json_file)

    # if "numbers" in data:
    #   data["numbers"] += batch_size  # Adds new numbers to the existing list
    # else:
    #   # If the key does not exist, create it
    #   data["numbers"] = batch_size

    # if text_prompt in data:
    #   data[text_prompt]['clip_score'] = data[text_prompt]['clip_score']*(data["numbers"]-batch_size) + clip_loss_1.L_N(traj_oc[-1]).detach().cpu().numpy().sum()*batch_size # Adds new numbers to the existing list
    #   data[text_prompt]['clip_score'] /= data["numbers"]
    #   data[text_prompt]['lpips'] = data[text_prompt]['lpips']*(data["numbers"]-batch_size) + lpips_f(traj_oc[-1], traj[-1]).detach().cpu().numpy().mean()
    #   data[text_prompt]['lpips'] /= data["numbers"]
    # else:
    #   # If the key does not exist, create it
    #   data[text_prompt] = {'clip_score':float(clip_loss_1.L_N(traj_oc[-1]).detach().cpu().numpy().sum()), 'lpips':lpips_f(traj_oc[-1], traj[-1]).detach().cpu().numpy().mean()}
      # data[text_prompt]['clip_score'] = clip_loss_1.L_N(traj_oc[-1]).detach().cpu().numpy().sum()
      # data[text_prompt]['lpips'] = lpips_f(traj_oc[-1], traj[-1]).detach().cpu().numpy().mean()

    # with open(file_path, 'w') as json_file:
    #   json.dump(data, json_file, indent=4)

    # print('oc similarity', id_loss(traj[-1], traj_oc[-1]))
    # print('gd similarity', id_loss(traj[-1], traj_gd[-1]))
  pass


def flowgrad_edit_batch(config, model_path, image_paths, text_prompt, output_dir):
   
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
  
  N = 100
  lr = 10
  beta = 1
  max_step = 10
  alpha = 0.7

  for img_path in tqdm(image_paths):
      target_dir = f'examples/{FLAGS.method}/{output_dir}'
      if img_path.startswith('examples/original'):
        opt_img_path = img_path.replace('examples/original', target_dir)
      else:
        opt_img_path = None

      # Load the image to edit
      image = get_img(img_path)  
      batch_size = 1

      original_img = image.to(config.device)
      clip_loss = clip_semantic_loss(text_prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)  
      
      t_s = time.time()
      latent = embed_to_latent(model_fn, scaler(original_img))
      traj = generate_traj(model_fn, latent, N=N)

      # Edit according to text prompt
      print(f'optimization starts: {img_path} -> {opt_img_path}')
      u_ind = [_ for _ in range(N)]
      u_opt = flowgrad_optimization_oc_d(latent, u_ind, model_fn, generate_traj, N=N, L_N=clip_loss.L_N,  number_of_iterations=max_step, alpha=lr,#first 3, second 2.75, third 2.5
                                  beta=beta) #first is 0.990, second is 0.9995, third is 0.995; first is 0.9925 third 0.995 last is 0.990

      traj_oc = generate_traj(model_fn, z0=latent, u=u_opt, N=N)

      if opt_img_path is not None:
        save_img(inverse_scaler(traj_oc[-1]), path=opt_img_path)

      with torch.no_grad():
        clip_loss_1 = clip_semantic_loss(text_prompt, original_img, config.device, alpha=1., inverse_scaler=inverse_scaler)  
        # id_loss = IDLoss(device=config.device)

        lpips_f = lpips.LPIPS(net='alex').to(config.device) # or 'vgg', 'squeeze'

        clip_loss = clip_loss_1.L_N(traj_oc[-1]).item()
        lpips_score = lpips_f(traj_oc[-1], traj[-1]).item()
        # id_loss = 1. - id_loss(traj[-1], traj_oc[-1]).detach().cpu().numpy()
        print(f'clip loss: {clip_loss:.4f}, lpips score: {lpips_score:.4f}, total time: {time.time() - t_s:.4f} s')
