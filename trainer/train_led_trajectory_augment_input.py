
from bdb import effective
import os
import time
import torch
import random
import numpy as np
import torch.nn as nn

from utils.config import Config
from utils.utils import print_log


from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate


from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel

import loralib as lora

import pdb
NUM_Tau = 5

class Trainer:
	def __init__(self, config):
		
		if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)
		self.device = torch.device('cuda') if config.cuda else torch.device('cpu')
		self.cfg = Config(config.cfg, config.info)
		
		# ------------------------- prepare train/test data loader -------------------------
		train_dset = NBADataset(
			obs_len=self.cfg.past_frames,
			pred_len=self.cfg.future_frames,
			training=True)

		self.train_loader = DataLoader(
			train_dset,
			batch_size=self.cfg.train_batch_size,
			shuffle=True,
			num_workers=4,
			collate_fn=seq_collate,
			pin_memory=True)
		
		test_dset = NBADataset(
			obs_len=self.cfg.past_frames,
			pred_len=self.cfg.future_frames,
			training=False)

		self.test_loader = DataLoader(
			test_dset,
			batch_size=self.cfg.test_batch_size,
			shuffle=False,
			num_workers=4,
			collate_fn=seq_collate,
			pin_memory=True)
		
		# data normalization parameters
		self.traj_mean = torch.FloatTensor(self.cfg.traj_mean).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)
		self.traj_scale = self.cfg.traj_scale

		# ------------------------- define diffusion parameters -------------------------
		self.n_steps = self.cfg.diffusion.steps # define total diffusion steps

		# make beta schedule and calculate the parameters used in denoising process.
		self.betas = self.make_beta_schedule(
			schedule=self.cfg.diffusion.beta_schedule, n_timesteps=self.n_steps, 
			start=self.cfg.diffusion.beta_start, end=self.cfg.diffusion.beta_end).cuda()
		
		self.alphas = 1 - self.betas
		self.alphas_prod = torch.cumprod(self.alphas, 0)
		self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
		self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)


		# ------------------------- define models -------------------------
		self.model = CoreDenoisingModel().cuda()
		# load pretrained models
		# model_cp = torch.load(self.cfg.pretrained_core_denoising_model, map_location='cpu') #if you need to retrain the diffusion model, keep it off
		# self.model.load_state_dict(model_cp['model_dict'])

		#changing the mode, with or without light and acceleration
		self.model_initializer = InitializationModel(t_h=10, d_h=6, t_f=20, d_f=2, k_pred=20).cuda()
		# self.model_initializer = InitializationModel(t_h=10, d_h=10, t_f=20, d_f=2, k_pred=20).cuda()
		# model_path = './results/led_augment/New_SinD_train_20epoch_withoulight_b_se/models/model_0020.p'
		# model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		# self.model_initializer.load_state_dict(model_dict)
		
		
		# ------------------------- lora models -------------------------
		#lora.mark_only_lora_as_trainable(self.model_initializer)
		#lora.mark_only_lora_as_trainable(self.model)

		# ------------------------- update parameters -------------------------
		self.opt = torch.optim.AdamW(self.model_initializer.parameters(), lr=config.learning_rate)
		self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)

		# ------------------------- prepare logs -------------------------
		self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
		self.print_model_param(self.model, name='Core Denoising Model')
		self.print_model_param(self.model_initializer, name='Initialization Model')

		# temporal reweight in the loss, it is not necessary.
		self.temporal_reweight = torch.FloatTensor([21 - i for i in range(1, 21)]).cuda().unsqueeze(0).unsqueeze(0) / 10


	def print_model_param(self, model: nn.Module, name: str = 'Model') -> None:
		'''
		Count the trainable/total parameters in `model`.
		'''
		total_num = sum(p.numel() for p in model.parameters())
		trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print_log("[{}] Trainable/Total: {}/{}".format(name, trainable_num, total_num), self.log)
		return None


	def make_beta_schedule(self, schedule: str = 'linear', 
			n_timesteps: int = 1000, 
			start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
		'''
		Make beta schedule.

		Parameters
		----
		schedule: str, in ['linear', 'quad', 'sigmoid'],
		n_timesteps: int, diffusion steps,
		start: float, beta start, `start<end`,
		end: float, beta end,

		Returns
		----
		betas: Tensor with the shape of (n_timesteps)

		'''
		if schedule == 'linear':
			betas = torch.linspace(start, end, n_timesteps)
		elif schedule == "quad":
			betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
		elif schedule == "sigmoid":
			betas = torch.linspace(-6, 6, n_timesteps)
			betas = torch.sigmoid(betas) * (end - start) + start
		return betas


	def extract(self, input, t, x):
		shape = x.shape
		out = torch.gather(input, 0, t.to(input.device))
		reshape = [t.shape[0]] + [1] * (len(shape) - 1)
		return out.reshape(*reshape)

	def noise_estimation_loss(self, x, y_0, mask):
		batch_size = x.shape[0]
		# Select a random step for each example
		t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,)).to(x.device)
		t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]
		# x0 multiplier
		a = self.extract(self.alphas_bar_sqrt, t, y_0)
		beta = self.extract(self.betas, t, y_0)
		# eps multiplier
		am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
		e = torch.randn_like(y_0)
		# model input
		y = y_0 * a + e * am1
		output = self.model(y, beta, x, mask)
		# batch_size, 20, 2
		return (e - output).square().mean()



	def p_sample(self, x, mask, cur_y, t):
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		# Factor to the model output
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		# Model output
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		eps_theta = self.model(cur_y, beta, x, mask)
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		# Generate z
		z = torch.randn_like(cur_y).to(x.device)
		# Fixed sigma
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z
		return (sample)
	
	def p_sample_accelerate(self, x, mask, cur_y, t):
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		# Factor to the model output
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		# Model output
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask)
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		# Generate z
		z = torch.randn_like(cur_y).to(x.device)
		# Fixed sigma
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z * 0.00001
		return (sample)



	def p_sample_loop(self, x, mask, shape):
		self.model.eval()
		prediction_total = torch.Tensor().cuda()
		for _ in range(20):
			cur_y = torch.randn(shape).to(x.device)
			for i in reversed(range(self.n_steps)):
				cur_y = self.p_sample(x, mask, cur_y, i)
			prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
		return prediction_total
	
	def p_sample_loop_mean(self, x, mask, loc):
		prediction_total = torch.Tensor().cuda()
		for loc_i in range(1):
			cur_y = loc
			for i in reversed(range(NUM_Tau)):
				cur_y = self.p_sample(x, mask, cur_y, i)
			prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
		return prediction_total

	def p_sample_loop_accelerate(self, x, mask, loc):
		'''
		Batch operation to accelerate the denoising process.

		x: [11, 10, 6]
		mask: [11, 11]
		cur_y: [11, 10, 20, 2]
		'''
		prediction_total = torch.Tensor().cuda()
		cur_y = loc[:, :10]
		for i in reversed(range(NUM_Tau)):
			cur_y = self.p_sample_accelerate(x, mask, cur_y, i)
		cur_y_ = loc[:, 10:]
		for i in reversed(range(NUM_Tau)):
			cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i)
		# shape: B=b*n, K=10, T, 2
		prediction_total = torch.cat((cur_y_, cur_y), dim=1)
		return prediction_total



	def fit(self):
		# Training loop
		start_time = time.time()
		#lora.mark_only_lora_as_trainable(self.model_initializer)
		#lora.mark_only_lora_as_trainable(self.model)
		for epoch in range(0, self.cfg.num_epochs):
			loss_total, loss_distance, loss_uncertainty = self._train_single_epoch(epoch)
			print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
				time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
				epoch, loss_total, loss_distance, loss_uncertainty), self.log)
			
			if (epoch + 1) % self.cfg.test_interval == 0:
				performance, samples = self._test_single_epoch()
				for time_i in range(4):
					print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(
						time_i+1, performance['ADE'][time_i]/samples,
						time_i+1, performance['FDE'][time_i]/samples), self.log)
				cp_path = self.cfg.model_path % (epoch + 1)
				model_cp = {'model_initializer_dict': self.model_initializer.state_dict()}
				#torch.save(lora.lora_state_dict(self.model_initializer), cp_path)
				#add saving path of diffusion model, retrain and save the diffusion model and initializer at the same time
				cp_path_diff = self.cfg.model_path_diff % (epoch + 1)
				model_cp_diff = {'model_dict': self.model.state_dict()}
				torch.save(model_cp, cp_path)
				torch.save(model_cp_diff, cp_path_diff)
			self.scheduler_model.step()
			end_time = time.time()
			print("cost_time_until_now:", end_time-start_time)



	def data_preprocess(self, data):
		"""
			pre_motion_3D: torch.Size([32, 11, 10, 2]), [batch_size, num_agent, past_frame, dimension]
			fut_motion_3D: torch.Size([32, 11, 20, 2])
			fut_motion_mask: torch.Size([32, 11, 20])
			pre_motion_mask: torch.Size([32, 11, 10])
			traj_scale: 1
			pred_mask: None
			seq: nba
		"""
		batch_size = data['pre_motion_3D'].shape[0]
		actor_num = data['pre_motion_3D'].shape[1]
		fut_traj_tmp = data['fut_motion_3D'].numpy()
		fut_traj = fut_traj_tmp[:,:,:,:-2]
		past_traj_tmp = data['pre_motion_3D'].numpy()
		past_traj = past_traj_tmp[:,:,:,:-2]
		mask = [True]*batch_size*actor_num
		effective_data_count = []
		for i in range(batch_size):
			e_count = 0
			for j in range(actor_num):
				# print(fut_traj.shape)
				# print(i)
				# print(j)
				# print(actor_num)
				id = i*actor_num+j
				if np.all(fut_traj[i,j,:,1]==1.2255*28/94):
				# if np.all(fut_traj[i,j,:,1]==1.2255):
					mask [id] = False
					e_count += 1
			effective_data_count.append(actor_num-e_count)
				#print(fut_traj[i,j,:,1])
					#actor_num -= 1
			#fut_traj = fut_traj[:,mask]
			#past_traj = past_traj[:,mask]
		fut_traj = torch.from_numpy(fut_traj)
		past_traj = torch.from_numpy(past_traj)
		past_traj_light = torch.from_numpy(past_traj_tmp[:,:,:,-2:])
		traj_mask = torch.zeros(mask.count(True), mask.count(True)).cuda()
		for i in range(batch_size):
			e_num = effective_data_count[i]
			e_accumulate = sum(effective_data_count[:i])
			traj_mask[e_accumulate:e_accumulate+e_num, e_accumulate:e_accumulate+e_num] = 1.
		# for i in range(batch_size):
		# 	traj_mask[i*actor_num:(i+1)*actor_num, i*actor_num:(i+1)*actor_num] = 1.

		initial_pos = past_traj.cuda()[:, :, -1:]
		# augment input: absolute position, relative position, velocity
		past_traj_abs = ((past_traj.cuda() - self.traj_mean)/self.traj_scale).contiguous().view(-1, 10, 2)
		past_traj_rel = ((past_traj.cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, 10, 2)
		#print(torch.zeros_like(past_traj_rel[:, -1:]).shape)
		past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1)
		past_traj_acc = torch.cat((past_traj_vel[:, 1:-1] - past_traj_vel[:, :-2], torch.zeros_like(past_traj_vel[:, -2:])), dim=1)
		#print(past_traj_vel.shape)
		light_signal = (past_traj_light.cuda()/3).contiguous().view(-1, 10, 2)
		# changing the mode, with or without light and acceleration
		# past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel, past_traj_acc, light_signal), dim=-1)
		past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
		#print(past_traj.shape)
		fut_traj = ((fut_traj.cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, 20, 2)
		fut_traj = fut_traj[mask]
		past_traj = past_traj[mask]

		return batch_size, traj_mask, past_traj, fut_traj, mask


	def _train_single_epoch(self, epoch):
		
		self.model.train()
		self.model_initializer.train()
		loss_total, loss_dt, loss_dc, count = 0, 0, 0, 0
		
		for data in self.train_loader:
			batch_size, traj_mask, past_traj, fut_traj, mask = self.data_preprocess(data)
			if mask.count(True) == 0:
				continue
			sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)

			sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			loc = sample_prediction + mean_estimation[:, None]
			
			#changing the mode, with or without light and acceleration
			# generated_y = self.p_sample_loop_accelerate(past_traj[:,:,:-4], traj_mask, loc)
			generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
			
			loss_dist = (	(generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1) 
								* 
							 self.temporal_reweight
						).mean(dim=-1).min(dim=1)[0].mean()
			loss_uncertainty = (torch.exp(-variance_estimation)
		       						*
								(generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2)) 
									+ 
								variance_estimation/20
								).mean()
			# print(variance_estimation.shape)
			# print(variance_estimation)
			loss = loss_dist*50 + loss_uncertainty
			loss_total += loss.item()
			loss_dt += loss_dist.item()*50
			loss_dc += loss_uncertainty.item()

			self.opt.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.)
			self.opt.step()
			count += 1
			if self.cfg.debug and count == 2:
				break

		return loss_total/count, loss_dt/count, loss_dc/count


	def _test_single_epoch(self):
		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0]}
		samples = 0
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)
		count = 0
		#print(self.test_loader)
		with torch.no_grad():
			for data in self.test_loader:
				batch_size, traj_mask, past_traj, fut_traj, mask = self.data_preprocess(data)
				#print(batch_size)
				if mask.count(True) == 0:
					continue
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]

				#changing the mode, with or without light and acceleration
				# pred_traj = self.p_sample_loop_accelerate(past_traj[:,:,:-4], traj_mask, loc)
				pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

				fut_traj = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
				for time_i in range(1, 5):
					ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()
				samples += distances.shape[0]
				count += 1
				# if count==100:
				# 	break
		return performance, samples


	def save_data(self):
		'''
		Save the visualization data.
		'''
		#model_path = './results/checkpoints/led_vis.p'
		# model_path = './results/checkpoints/led_new.p'
		# model_path = './results/led_augment/test/models/model_0100.p'
		# model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		# self.model_initializer.load_state_dict(model_dict)
		model_path = './results/led_augment/SinD_train_light_600_10epoch/models/model_0010.p'
		model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		self.model_initializer.load_state_dict(model_dict)
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)
		root_path = './visualization/data/light_600_10epoch/'
		
		fut_traj_all = torch.empty(1,20,4).cuda()
		past_traj_all = torch.empty(1,10,4).cuda()
		pred_traj_all = torch.empty(1, 20, 20, 2).cuda()
		pred_mean_all = torch.empty(1,1,20,2).cuda()
		sample_prediction_all = torch.empty(1,20,20,2).cuda()
		mean_estimation_all = torch.empty(1,20,2).cuda()
		variance_estimation_all = torch.empty(1,1).cuda()
		agent_number = []
		with torch.no_grad():
			for data in self.test_loader:
				_, traj_mask, past_traj, _ , mask= self.data_preprocess(data)
				fut_traj_tmp = data['fut_motion_3D']
				past_traj_tmp = data['pre_motion_3D']
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)


				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]

				#changing the mode, with or without light and acceleration
				pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
				pred_mean = self.p_sample_loop_mean(past_traj, traj_mask, mean_estimation)
				# pred_traj = self.p_sample_loop_accelerate(past_traj[:,:,:-4], traj_mask, loc)
				# pred_mean = self.p_sample_loop_mean(past_traj[:,:,:-4], traj_mask, mean_estimation)
				
				fut_traj_tmp = fut_traj_tmp[:,mask].squeeze(0).cuda()
				past_traj_tmp = past_traj_tmp[:,mask].squeeze(0).cuda()
				pred_traj = pred_traj[mask].cuda()
				pred_mean = pred_mean[mask].cuda()
				sample_prediction = sample_prediction[mask].cuda()
				mean_estimation = mean_estimation[mask].cuda()
				variance_estimation = variance_estimation[mask].cuda()
				agent_number.append(len(fut_traj_tmp))

				fut_traj_all = torch.cat((fut_traj_all,fut_traj_tmp),dim = 0)
				past_traj_all = torch.cat((past_traj_all,past_traj_tmp),dim = 0)
				pred_traj_all = torch.cat((pred_traj_all,pred_traj),dim = 0)
				pred_mean_all = torch.cat((pred_mean_all,pred_mean),dim = 0)
				sample_prediction_all = torch.cat((sample_prediction_all,sample_prediction),dim = 0)
				mean_estimation_all = torch.cat((mean_estimation_all,mean_estimation),dim = 0)
				variance_estimation_all = torch.cat((variance_estimation_all,variance_estimation),dim = 0)
				
				

			torch.save(sample_prediction_all[1:], root_path+'p_var.pt')
			torch.save(mean_estimation_all[1:], root_path+'p_mean.pt')
			torch.save(variance_estimation_all[1:], root_path+'p_sigma.pt')
			torch.save(past_traj_all[1:], root_path+'past.pt')
			torch.save(fut_traj_all[1:], root_path+'future.pt')
			torch.save(pred_traj_all[1:], root_path+'prediction.pt')
			torch.save(pred_mean_all[1:], root_path+'p_mean_denoise.pt')
			torch.save(agent_number, root_path+'agent_number.pt')

				#raise ValueError



	def test_single_model(self):
		#change test model here
		# model_path = './results/checkpoints/led_new.p'
		# model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		# self.model_initializer.load_state_dict(model_dict, strict=False)
		# model_path_lora = './results/checkpoints/led_new.p'
		# model_dict_lora = torch.load(model_path_lora, map_location=torch.device('cpu'))
		# self.model_initializer.load_state_dict(model_dict_lora, strict=False)
		# model_path = './results/led_augment/test/models/model_0100.p'
		# model_path = './results/led_augment/SinD_train_light_600_10epoch/models/model_0010.p'
		model_path = './results/led_augment/New_diff_SinD_train_100epoch_withoutlight_t2/models/model_0100.p'
		model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		self.model_initializer.load_state_dict(model_dict)

		#if you need to load the retrained diffusion model, keep the following on
		model_diff_path = './results/led_augment/New_diff_SinD_train_100epoch_withoutlight_t2/models/model_diff_0100.p'
		model_diff_dict = torch.load(model_diff_path, map_location=torch.device('cpu'))['model_dict']
		self.model.load_state_dict(model_diff_dict)
		
		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0]}
		samples = 0
		print_log(model_path, log=self.log)
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)
		count = 0
		#print(len(self.test_loader))
		with torch.no_grad():
			for data in self.test_loader:
				batch_size, traj_mask, past_traj, fut_traj, mask= self.data_preprocess(data)
				if mask.count(True) == 0:
					continue
				#print(batch_size)
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]

				#changing the mode, with or without light and acceleration
				pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
				# pred_traj = self.p_sample_loop_accelerate(past_traj[:,:,:-4], traj_mask, loc)

				#print(fut_traj.shape)
				fut_traj = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
				#print(mask)
				# fut_traj = fut_traj[mask]
				# pred_traj = pred_traj[mask]
				#print(fut_traj.shape)
				#print(pred_traj.shape)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
				for time_i in range(1, 5):
					ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()
				samples += distances.shape[0]
				count += 1
					# if count==2:
					# 	break
		for time_i in range(4):
			print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(time_i+1, performance['ADE'][time_i]/samples, \
				time_i+1, performance['FDE'][time_i]/samples), log=self.log)
		
	