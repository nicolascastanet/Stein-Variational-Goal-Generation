# 1. Imports
import sys
sys.path.append("./")
from mrl.import_all import *
from mrl.modules.train import debug_vectorized_experience
from experiments.mega.make_env import make_env
from experiments.mega.test_tensor import tensor_point_maze, tensor_ant_maze, tensor_maze_2, tensor_maze_square_c2, tensor_point_maze_random, tensor_ant_maze_random
import time
import os
import gym
import numpy as np
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import cProfile
import re
import pstats, io
from pstats import SortKey

def grid_eval_point_maze(agent, args, num_ep=20,nb_grid=11,nb_sample=10):
  agent.eval_mode()

  h = 0.4
  if 'pointmaze' in args.env.lower():
    #goal_test_tensor = tensor_point_maze(density=h)
    goal_test_tensor = tensor_point_maze_random(agent=agent,d=nb_grid,nb_sample=nb_sample)
  elif 'antmaze' in args.env.lower():
    goal_test_tensor = tensor_ant_maze_random(agent=agent,nb_sample=4)
  else:
    raise NotImplementedError

  l_goal,_ = goal_test_tensor.shape
  output = np.zeros((l_goal,1))

  for i in range(l_goal):
    if i % 100 == 0:
      print("-----\nTest goal success number :", i)
    output[i] = np.mean(agent.eval(num_episodes=num_ep,goal=goal_test_tensor[i]).is_successes)

  return output
  

# 2. Get default config and update any defaults (this automatically updates the argparse defaults)
config = protoge_config()
# 3. Make changes to the argparse below

def main(args):

  # 4. Update the config with args, and make the agent name. 
  if args.num_envs is None:
    import multiprocessing as mp
    args.num_envs = max(mp.cpu_count() - 1, 1)

  merge_args_into_config(args, config)


  
  if config.gamma < 1.: config.clip_target_range = (np.round(-(1 / (1-config.gamma)), 2), 0.)
  if config.gamma == 1: config.clip_target_range = (np.round(- args.env_max_step - 5, 2), 0.)

  if args.sparse_reward_shaping:
    config.clip_target_range = (-np.inf, np.inf)

  config.agent_name = make_agent_name(config, ['env','alg','her','seed','tb','ag_curiosity','first_visit_succ','beta','svgd_steps','oe_part','succ_oe'], prefix=args.prefix)
  
  # MEGA Old config
  #config.agent_name = make_agent_name(config, ['env','alg','her','layers','seed','tb','ag_curiosity','eexplore','first_visit_succ', 'dg_score_multiplier','alpha'], prefix=args.prefix)
  # Directly put the agent name to load existing agent
  #config.agent_name = """proto_env-pointmaze_alg-DDPG_herrfaab_1_4_0_1_4_layer-(512, 512, 512)_seed111_tb-MEGA_ag_cu-minkde_eexpl0.1_first-True_dg_sc1.0_alpha--1.0"""

  # 5. Setup / add basic modules to the config
  config.update(
      dict(
          trainer=StandardTrain(),
          evaluation=EpisodicEval(),
          policy=ActorPolicy(),
          logger=Logger(),
          state_normalizer=Normalizer(MeanStdNormalizer()),
          replay=OnlineHERBuffer(),
      ))


  config.prioritized_mode = args.prioritized_mode
  if config.prioritized_mode == 'mep':
    config.prioritized_replay = EntropyPrioritizedOnlineHERBuffer()

  if not args.no_ag_kde:
    config.ag_kde = RawKernelDensity('ag', optimize_every=1, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
    #config.torch_kde = TorchKDE('ag', samples=100, bw=args.bandwidth)
    #config.gaussian_mixture = GaussianMixture(M=30,sparsity=5, optimize_every=4000, samples=5000, epoch=300)
    #config.mggd = GenGaussianPytorch(input_shape=2, samples=10000)
  
  if args.ag_curiosity is not None:
    if not args.no_ag_kde:
      config.dg_kde = RawKernelDensity('dg', optimize_every=500, samples=10000, kernel='tophat', bandwidth = 0.2)
      config.ag_kde_tophat = RawKernelDensity('ag', optimize_every=100, samples=10000, kernel='tophat', bandwidth = 0.2, tag='_tophat')
    if args.transition_to_dg:
      config.alpha_curiosity = CuriosityAlphaMixtureModule()
    if 'rnd' in args.ag_curiosity:
      config.ag_rnd = RandomNetworkDensity('ag')
    if 'flow' in args.ag_curiosity:
      config.ag_flow = FlowDensity('ag')

    use_qcutoff = not args.no_cutoff

    if args.ag_curiosity == 'minq':
      config.ag_curiosity = QAchievedGoalCuriosity(max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randq':
      config.ag_curiosity = QAchievedGoalCuriosity(max_steps = args.env_max_step, randomize=True, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'minkde':
      config.ag_curiosity = DensityAchievedGoalCuriosity(max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'minrnd':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_rnd', max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'minflow':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_flow', max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randkde':
      config.ag_curiosity = DensityAchievedGoalCuriosity(alpha = args.alpha, max_steps = args.env_max_step, randomize=True, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randrnd':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_rnd', alpha = args.alpha, max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randflow':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_flow', alpha = args.alpha, max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'goaldisc':

      if args.disc_classifier.lower() == 'nn':
        config.success_predictor = NNPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe, k_steps=args.k_steps)
      elif args.disc_classifier.lower() in {'svm', 'knn', 'trees'}:
        config.success_predictor = SklearnPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe, k_steps=args.k_steps)
      else:
        raise NotImplementedError

      config.ag_curiosity = SuccessAchievedGoalCuriosity(max_steps=args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'goaldiscsample':

      if args.disc_classifier.lower() == 'nn':
        config.success_predictor = NNPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe, k_steps=args.k_steps)
      elif args.disc_classifier.lower() in {'svm', 'knn', 'trees'}:
        config.success_predictor = SklearnPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe, k_steps=args.k_steps)
      else:
        raise NotImplementedError

      if args.mc_drop:
        mc = True
      else:
        mc = False
      config.ag_curiosity = SuccessAchievedGoalCuriositySample(beta = args.beta, mc=False, max_steps=args.env_max_step, sample=True, num_sampled_ags=args.num_sampled_ags, get_last_ags=args.get_last_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)

    elif args.ag_curiosity =='thompson':
      config.success_predictor = NNPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe, k_steps=args.k_steps)
      config.ag_curiosity = ThompsonSampling(beta = args.beta, mc=False, max_steps=args.env_max_step, sample=True, num_sampled_ags=args.num_sampled_ags, get_last_ags=args.get_last_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)

    elif args.ag_curiosity =='svgg':

      test = args.use_prior
      if args.use_prior:
        prior_type = 'ocsvm'
        config.ocsvm = OCSVMdensity('ag', optimize_every=args.oe_prior, samples=5000, gamma=args.gamma_ocsvm,nu=0.01)
      else:
        prior_type=None

      if args.disc_classifier.lower() == 'nn':
        config.success_predictor = NNPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe, k_steps=args.k_steps, goal_pred=args.goal_predict)
      elif args.disc_classifier.lower() in {'svm', 'knn', 'trees'}:
        config.success_predictor = SklearnPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe, k_steps=args.k_steps)
      else:
        raise NotImplementedError
                                                                                                                                                                       
      config.ag_curiosity = SvgdEntropy(beta = args.beta, nb_particles=args.nb_particles, epoch=args.svgd_steps,annealed=args.annealed_trick ,slope=args.slope,oe_svgd=args.oe_part,grad_clip=args.grad_clip_svgd,prior_type=prior_type,proj=args.use_proj,use_prior=args.use_prior,svgd_lr=args.svgd_lr,proj_coord=args.l2_proj,sigma=args.sigma,alpha_beta=args.alpha_beta, max_steps=args.env_max_step, sample=True, num_sampled_ags=args.num_sampled_ags, get_last_ags=args.get_last_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
      
    elif args.ag_curiosity =='svgd_only_prior':
      config.ocsvm = OCSVMdensity('ag', optimize_every=args.oe_prior, samples=5000, gamma=args.gamma_ocsvm,nu=0.01)
      config.ag_curiosity = SvgdEntropy(beta = args.beta, nb_particles=args.nb_particles, epoch=args.svgd_steps,annealed=args.annealed_trick ,slope=args.slope, only_prior=True, use_prior=True,prior_type='ocsvm',proj=False,max_steps=args.env_max_step, sample=True, num_sampled_ags=args.num_sampled_ags, get_last_ags=args.get_last_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    
    elif args.ag_curiosity == 'goal_gan':
      config.success_predictor = NNPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe, k_steps=args.k_steps)
      config.ag_curiosity = GoalGAN(max_steps=args.env_max_step, sample=True, num_sampled_ags=args.num_sampled_ags, get_last_ags=args.get_last_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent, noise_dim=args.gan_noise_dim)
      config.gan_trainer = GoalGanTrainer(noise_dim=args.gan_noise_dim, k_steps=args.k_steps)

    elif args.ag_curiosity == 'lazy':
      if args.lazy_distrib:
        r = False; s = True
      else:
        r = True; s=False
      config.ag_curiosity = LazyRelabelGoals(beta=args.beta, num_ep = args.num_ep_eval ,rand=r,sample_bis=s, max_steps=args.env_max_step, num_sampled_ags=args.num_sampled_ags, get_last_ags=args.get_last_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
 
  
    elif args.ag_curiosity == 'uncertainty':
      config.ag_curiosity = UncertaintyCuriosity(beta = args.beta, MC_samples=args.num_MC_samples, mode=args.uncertainty_measure, max_steps=args.env_max_step, sample=True, get_last_ags=args.get_last_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'entropygainscore':
      config.bg_kde = RawKernelDensity('bg', optimize_every=args.env_max_step, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
      config.bgag_kde = RawJointKernelDensity(['bg','ag'], optimize_every=args.env_max_step, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
      config.ag_curiosity = EntropyGainScoringGoalCuriosity(max_steps=args.env_max_step, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    else:
      raise NotImplementedError

  if args.noise_type.lower() == 'gaussian': noise_type = GaussianProcess
  if args.noise_type.lower() == 'ou': noise_type = OrnsteinUhlenbeckProcess
  config.action_noise = ContinuousActionNoise(noise_type, std=ConstantSchedule(args.action_noise))

  if args.alg.lower() == 'ddpg': 
    config.algorithm = DDPG()
  elif args.alg.lower() == 'td3':
    config.algorithm = TD3()
    config.target_network_update_freq *= 2
  elif args.alg.lower() == 'sac':
    config.algorithm = SAC()
    config.target_network_update_freq *= 2
  elif args.alg.lower() == 'dqn': 
    config.algorithm = DQN()
    config.policy = QValuePolicy()
    config.qvalue_lr = config.critic_lr
    config.qvalue_weight_decay = config.actor_weight_decay
    config.double_q = True
    config.random_action_prob = LinearSchedule(1.0, config.eexplore, 1e5)
  else:
    raise NotImplementedError

  # 6. Setup / add the environments and networks (which depend on the environment) to the config
  env, eval_env = make_env(args)
  if args.first_visit_done:
    env1, eval_env1 = env, eval_env
    env = lambda: FirstVisitDoneWrapper(env1())
    eval_env = lambda: FirstVisitDoneWrapper(eval_env1())
  if args.first_visit_succ:
    config.first_visit_succ = True

  config.train_env = EnvModule(env, num_envs=args.num_envs, seed=args.seed)
  config.eval_env = EnvModule(eval_env, num_envs=args.num_eval_envs, name='eval_env', seed=args.seed + 1138)

  e = config.eval_env
  if args.alg.lower() == 'dqn':
    config.qvalue = PytorchModel('qvalue', lambda: Critic(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), e.action_dim))
  else:
    config.actor = PytorchModel('actor',
                                lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), e.action_dim, e.max_action))
    config.critic = PytorchModel('critic',
                                lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), 1))
    if args.alg.lower() in {'td3','sac'}:
      config.critic2 = PytorchModel('critic2',
        lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), 1))
    
    a_dim=e.action_dim
    print("action dim:",a_dim)

    if args.alg.lower() == 'sac':
      del config.actor
      del config.action_noise
      del config.policy

      config.module_policy = StochasticActorPolicy()
      config.actor = PytorchModel(
      'actor', lambda: StochasticActor(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), 
        e.action_dim, e.max_action, log_std_bounds = (-20, 2)))

  if args.goal_predict:
    input_dim = e.goal_dim
  else:
    input_dim = e.state_dim

  if args.ag_curiosity == 'goal_gan':
    config.gan_discriminator = PytorchModel('gan_discriminator', lambda: Critic(FCBody(input_dim + e.goal_dim, args.sp_layers, nn.LayerNorm, make_activ(config.activ)), 1))
    config.gan_generator = PytorchModel('gan_generator', lambda: Critic(FCBody(args.gan_noise_dim, args.sp_layers, nn.LayerNorm, make_activ(config.activ)), e.goal_dim))
      

  if args.ag_curiosity in {'goaldisc', 'goaldiscsample','thompson','svgd', 'goal_gan'}:
    if args.disc_classifier.lower() == 'nn':
      if args.mc_drop:
        config.goal_discriminator = PytorchModel('goal_discriminator', lambda: Critic(MCdropBody(input_dim + e.goal_dim, args.sp_layers, nn.LayerNorm, make_activ(config.activ), proba=args.drop_prob_MC), 1))
      else:  
        config.goal_discriminator = PytorchModel('goal_discriminator', lambda: Critic(FCBody(input_dim + e.goal_dim, args.sp_layers, nn.LayerNorm, make_activ(config.activ)), 1))

    elif args.disc_classifier.lower() == 'knn':
      config.goal_discriminator = SklearnModel('goal_discriminator', lambda: KNeighborsClassifier(n_neighbors=20))  

    elif args.disc_classifier.lower() == 'svm':
      config.goal_discriminator = SklearnModel('goal_discriminator', lambda: SVC(C=1, kernel = 'rbf',degree=2, gamma='auto', class_weight='balanced', probability=True))

    elif args.disc_classifier.lower() == 'trees':
      config.goal_discriminator = SklearnModel('goal_discriminator', lambda: RandomForestClassifier(n_estimators = 200, max_depth=3, random_state=0, class_weight='balanced')) 

    else:
      raise NotImplementedError
  """
  elif args.ag_curiosity in {'uncertainty'}:
    config.critic = PytorchModel('critic',
                                lambda: Critic(MCdropBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm, make_activ(config.activ), proba=args.drop_prob_MC), 1))"""

  #import ipdb;ipdb.set_trace()

  if args.reward_module == 'env':
    config.goal_reward = GoalEnvReward()
  elif args.reward_module == 'intrinsic':
    config.goal_reward = NeighborReward()
    config.neighbor_embedding_network = PytorchModel('neighbor_embedding_network',
                                                     lambda: FCBody(e.goal_dim, (256, 256)))
  else:
    raise ValueError('Unsupported reward module: {}'.format(args.reward_module))

  if config.eval_env.goal_env:
    if not (args.first_visit_done or args.first_visit_succ):
      config.never_done = True  # NOTE: This is important in the standard Goal environments, which are never done


  # 7. Make the agent and run the training loop.
  agent = mrl.config_to_agent(config)


  if args.visualize_trained_agent:
    print("Loading agent at epoch {}".format(0))
    agent.load('checkpoint')
    
    if args.intrinsic_visualization:
      agent.eval_mode()
      agent.train(10000, render=True, dont_optimize=True)

    else:
      agent.eval_mode()
      env = agent.eval_env

      for _ in range(10000):
        print("NEW EPISODE")
        state = env.reset()
        env.render()
        done = False
        while not done:
          time.sleep(0.02)
          action = agent.policy(state)
          state, reward, done, info = env.step(action)
          env.render()
          print(reward[0])

  elif args.grid_eval:
    out = grid_eval_point_maze(agent, args, num_ep=20,nb_sample=10)
    np.save(agent.agent_folder + '/grid_succ_0', out)

  elif args.get_trajectories:

    goal_test_tensor = np.array([[5,5]
                                    ,[2,3]
                                    ,[5,2]
                                    ,[0,5]
                                    ,[5,0]
                                    ,[3,4]
                                    ,[1,2]
                                     ])
    l_goal,_ = goal_test_tensor.shape
    trajs = []
    success = []

    for i in range(l_goal):
      res = agent.eval(num_episodes=10,goal=goal_test_tensor[i])
      trajs.append(res.states)
      success.append(res.is_successes)

    traj_folder = agent.agent_folder + "/test_traj/"
    if not os.path.exists(traj_folder):
      print("Creating traj folder ...\n")
      os.mkdir(traj_folder)
    np.save(traj_folder + '/success', np.array(success))
    np.save(traj_folder + '/trajs', np.array(trajs))
    np.save(traj_folder + '/test_goals', goal_test_tensor)

  else:
    ag_buffer = agent.replay_buffer.buffer.BUFF.buffer_ag
    bg_buffer = agent.replay_buffer.buffer.BUFF.buffer_bg

    # EVALUATE
    res = np.mean(agent.eval(num_episodes=30).rewards)
    agent.logger.log_color('Initial test reward (30 eps):', '{:.2f}'.format(res))

    for epoch in range(int(args.max_steps // args.epoch_len)):
      t = time.time()

      if args.new_maze_type is not None and epoch == args.change_env_epoch:

        print("change env !")

        env, eval_env = make_env(args,change_env=True)
        if args.first_visit_done:
          env1, eval_env1 = env, eval_env
          env = lambda: FirstVisitDoneWrapper(env1())
          eval_env = lambda: FirstVisitDoneWrapper(eval_env1())
        
        agent.env = EnvModule(env, num_envs=args.num_envs, seed=args.seed)
        agent.eval_env = EnvModule(eval_env, num_envs=args.num_eval_envs, name='eval_env', seed=args.seed + 1138)
        #agent.env.env.dummy_env.env.change_maze_type(args.new_maze_type)
        #agent.eval_env.env.dummy_env.env.change_maze_type(args.new_maze_type)

        
      #pr = cProfile.Profile()
      #pr.enable()
    
      agent.train(num_steps=args.epoch_len)

      #pr.disable()
      #s = io.StringIO()
      #sortby = SortKey.CUMULATIVE
      #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
      #ps.print_stats()
      #print(s.getvalue())
      
      # VIZUALIZE GOALS
      if args.save_embeddings:
        sample_idxs = np.random.choice(len(ag_buffer), size=min(len(ag_buffer), args.epoch_len), replace=False)
        last_idxs = np.arange(max(0, len(ag_buffer)-args.epoch_len), len(ag_buffer))
        agent.logger.add_np_embedding('rand_ags', ag_buffer.get_batch(sample_idxs), upper_tag='goals')
        agent.logger.add_np_embedding('last_ags', ag_buffer.get_batch(last_idxs), upper_tag='goals')
        agent.logger.add_np_embedding('last_bgs', bg_buffer.get_batch(last_idxs), upper_tag='goals')

      
      # EVALUATE
      res = np.mean(agent.eval(num_episodes=30).rewards)
      agent.logger.log_color('Test reward (30 eps):', '{:.2f}'.format(res))
      agent.logger.log_color('Epoch time:', '{:.2f}'.format(time.time() - t), color='yellow')

      if args.grid_eval_periodic and epoch % args.grid_eval_freq == 0 and epoch != 0: # every 100 000 steps for now
        out = grid_eval_point_maze(agent, args, num_ep=20,nb_sample=10)
        #out = grid_eval_point_maze(agent, args, num_ep=20)
        #out = agent.eval.success_coverage()
        agent.logger.add_np_embedding('success', out, upper_tag = 'grid_eval')
        success_coverage = out.sum()/len(out)
        agent.logger.add_scalar('Test/Success_coverage', success_coverage)

      print("Saving agent at epoch {}".format(epoch))
      agent.save('checkpoint')


# 3. Declare args for modules (also parent_folder is required!)
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Train DDPG", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--parent_folder', default='/tmp/test_mega', type=str, help='where to save progress')
  parser.add_argument('--device',default='cpu',type=str, help='device')
  parser.add_argument('--prefix', type=str, default='proto', help='Prefix for agent name (subfolder where it is saved)')
  parser.add_argument('--env', default="FetchPush-v1", type=str, help="gym environment")
  parser.add_argument('--max_steps', default=5000000, type=int, help="maximum number of training steps")
  parser.add_argument('--alg', default='DDPG', type=str, help='algorithm to use (DDPG or TD3)')
  parser.add_argument(
      '--layers', nargs='+', default=(512,512,512), type=int, help='sizes of layers for actor/critic networks')    
  parser.add_argument('--noise_type', default='Gaussian', type=str, help='type of action noise (Gaussian or OU)')
  parser.add_argument('--tb', default='', type=str, help='a tag for the agent name / tensorboard')
  parser.add_argument('--epoch_len', default=5000, type=int, help='number of steps between evals')
  parser.add_argument('--num_envs', default=None, type=int, help='number of envs')
  parser.add_argument('--her', default='rfaab_1_4_3_1_1', type=str, help='HER Relabeling strategy')
  parser.add_argument('--inverseRL', default=False, type=bool, help='Use Inverse RL (IRL) for relabeling')

  # Make env args
  parser.add_argument('--maze_type', default='square_large_0', type=str, help='choose maze for point_maze environment')
  parser.add_argument('--eval_env', default='', type=str, help='evaluation environment')
  parser.add_argument('--test_with_internal', default=True, type=bool, help='test with internal reward fn')
  parser.add_argument('--reward_mode', default=0, type=int, help='reward mode')
  parser.add_argument('--env_max_step', default=50, type=int, help='max_steps_env_environment')
  parser.add_argument('--per_dim_threshold', default='0.', type=str, help='per_dim_threshold')
  parser.add_argument('--hard', action='store_true', help='hard mode: all goals are high up in the air')
  parser.add_argument('--pp_in_air_percentage', default=0.5, type=float, help='sets in air percentage for fetch pick place')
  parser.add_argument('--pp_min_air', default=0.2, type=float, help='sets the minimum height in the air for fetch pick place when in hard mode')
  parser.add_argument('--pp_max_air', default=0.45, type=float, help='sets the maximum height in the air for fetch pick place')
  parser.add_argument('--train_dt', default=0., type=float, help='training distance threshold')
  parser.add_argument('--slow_factor', default=1., type=float, help='slow factor for moat environment; lower is slower. ')

  # SVGG
  parser.add_argument('--svgd_steps', default=1, type=int, help='num of gradient step for particle learning in svgd')
  parser.add_argument('--nb_particles', default=100, type=int, help='num of particles for svgd')
  parser.add_argument('--sigma', default=1.0, type=float, help='sigma for RBF kernel, if None : adaptatif heuristic')
  parser.add_argument('--annealed_trick', default=False, type=bool, help='use annealed trick or not for svgd')
  parser.add_argument('--slope', default=3, type=int, help='slope for annealed trick in svgd')
  parser.add_argument('--oe_part', default=20, type=int, help='optimize every k steps particles with SVGD')
  parser.add_argument('--beta', default=10.0, type=float, help='temperature factor for goal sampling (0 -> Uniform)')
  parser.add_argument('--grad_clip_svgd', default=1000, type=float, help='gradient threshold for wich we prevent particle attraction')
  parser.add_argument('--svgd_lr', default=1e-1, type=float, help='Learning rate for SVGD')
  parser.add_argument('--l2_proj', default=[2,2,6], type=list, help='particles L2 projection [x,y,rayon]')
  parser.add_argument('--use_proj', default=False, type=bool, help='use L2 projection')
  parser.add_argument('--use_prior', default=False, type=bool, help='use prior distribution in SVGD algorithm')
  parser.add_argument('--oe_prior', default=4000, type=int, help='optimize prior every k steps')
  parser.add_argument('--gamma_ocsvm', default=0.5, type=float, help='gamma of rbf kernel for OCSVM')
  parser.add_argument('--alpha_beta',nargs='+', default=None, type=float, help='alpha / beta args for beta distribution when used in SVGG')
  
  
  # Success prediction args
  parser.add_argument(
      '--sp_layers', nargs='+', default=(64,64), type=int, help='sizes of layers for success predictor network networks')
  parser.add_argument('--k_steps', default=60, type=int, help='k steps for success prediction optimization')
  parser.add_argument('--disc_classifier', default='NN', type=str, help='type of Goal Discriminator classifier in {NN, SVM, Knn, Trees}')
  parser.add_argument('--succ_bs', default=60, type=int, help='success predictor batch size')
  parser.add_argument('--succ_hl', default=300, type=int, help='success predictor history length')
  parser.add_argument('--succ_oe', default=4000, type=int, help='success predictor optimize every')
  parser.add_argument('--goal_predict', default=False, type=bool, help='goal or state for succ pred input')


  

  # Goal GAN
  parser.add_argument('--gan_noise_dim', default=4, type=int, help='dimension if noise input in GAN training')
  
 
  # Other args
  parser.add_argument('--first_visit_succ', action='store_true', help='Episodes are successful on first visit (soft termination).')
  parser.add_argument('--first_visit_done', action='store_true', help='Episode terminates upon goal achievement (hard termination).')
  parser.add_argument('--ag_curiosity', default=None, help='the AG Curiosity model to use: {minq, randq, minkde, goaldisc, goaldiscsample}')
  parser.add_argument('--bandwidth', default=0.1, type=float, help='bandwidth for KDE curiosity')
  parser.add_argument('--kde_kernel', default='gaussian', type=str, help='kernel for KDE curiosity')
  parser.add_argument('--num_sampled_ags', default=100, type=int, help='number of ag candidates sampled for curiosity')
  parser.add_argument('--alpha', default=-1.0, type=float, help='Skewing parameter on the empirical achieved goal distribution. Default: -1.0')
  parser.add_argument('--drop_prob_MC', default=0.2, type=float, help='dropout proba for uncertainty measure')
  parser.add_argument('--mc_drop', action='store_true', help="use drop net or not")
  parser.add_argument('--num_MC_samples', default=100, type=int, help='num sample use in forward passe to compute uncertainty with dropout')
  parser.add_argument('--uncertainty_measure', default='var', type=str, help='type of uncertainty in (var-ratios, entropy, mut-inf)')
  parser.add_argument('--get_last_ags', default=False, type=bool, help='If true get the last ags to get the behavior goal, False : random ags')
  parser.add_argument('--reward_module', default='env', type=str, help='Reward to use (env or intrinsic)')
  parser.add_argument('--save_embeddings', action='store_true', help='save ag embeddings during training?')
  parser.add_argument('--num_ep_eval', default=2, type=int, help='num eval episode for relabeling in laisy training curiosity')
  parser.add_argument('--lazy_distrib', default=False, type=bool, help='Softmax dsitrib or not for lazy curiosity')
  parser.add_argument('--ag_pred_ehl', default=5, type=int, help='achieved goal predictor number of timesteps from end to consider in episode')
  parser.add_argument('--transition_to_dg', action='store_true', help='transition to the dg distribution?')
  parser.add_argument('--no_cutoff', action='store_true', help="don't use the q cutoff for curiosity")
  parser.add_argument('--visualize_trained_agent', action='store_true', help="visualize the trained agent")
  parser.add_argument('--grid_eval', action='store_true', help='average success of the agent over grid space')
  parser.add_argument('--grid_eval_periodic', action='store_true', help='periodicly eval success coverage')
  parser.add_argument('--grid_eval_freq', default=40, type=int, help='num of epoch between grid evaluation')
  parser.add_argument('--get_trajectories', action='store_true', help='get states for specific goals')
  parser.add_argument('--intrinsic_visualization', action='store_true', help="if visualized agent should act intrinsically; requires saved replay buffer!")
  parser.add_argument('--keep_dg_percent', default=-1e-1, type=float, help='Percentage of time to keep desired goals')
  parser.add_argument('--prioritized_mode', default='none', type=str, help='Modes for prioritized replay: none, mep (default: none)')
  parser.add_argument('--no_ag_kde', action='store_true', help="don't track ag kde")
  #parser.add_argument('--change_env', action='store_true', help="change env during training")
  parser.add_argument('--new_maze_type', default=None, type=str, help="new maze type after change")
  parser.add_argument('--change_env_epoch', default=50, type=int, help="epoch when to change environment")

  parser = add_config_args(parser, config)
  args = parser.parse_args()

  import subprocess, sys
  args.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  main(args)
