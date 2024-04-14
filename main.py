'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
要点：1.输出随机策略，泛化性强（和之前PPO的思路一样，其实是正确的）
2.基于最大熵，综合评价奖励值与动作多样性（-log p(a\s)表示熵函数，单个动作概率越大，熵函数得分越低；为了避免智能体薅熵函数分数（熵均贫化），
可以动态改变熵函数系数（温度系数），先大后小，实现初期注重探索后期注重贪婪
3.【重点】随机策略同样是网络生成方差+均值，和“以为PPO能做的”一样，但是必须用重参数化，即不直接从策略分布中取样，而是从标准正态分布
N(0,1)中取样i，与网络生成的方差+均值mu,sigma得到实际动作a=mu+sigma*i，这样保留了参数本身，才能利用链式法则求出loss相对参数本身的梯度；
如果直接取样a，a与参数mu和sigma没有关系，根本没法求相对mu和sigma的梯度；之前隐隐觉得之前的PPO算法中间隔了个正态分布所以求梯度这一步存在问题其实是对的...
4.目前SAC实现的算法（openAI和作者本人的）都用了正态分布替代多模Q函数，如果想用多模Q函数需要用网络实现SVGD取样方法拟合多模Q函数（也是发明人在原论文中用的方法(Soft Q-Learning不是SAC））
'''

import datetime
import numpy as np
import itertools
import torch.nn as nn
from sac import SAC
#from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import env 
import time
import matplotlib.pyplot as plt
import numpy as np

# 字典形式存储全部参数
args={'policy':"Gaussian", # Policy Type: Gaussian | Deterministic (default: Gaussian)
        'eval':True, # Evaluates a policy a policy every 10 episode (default: True)
        'gamma':0.99, # discount factor for reward (default: 0.99)
        'tau':0.1, # target smoothing coefficient(τ) (default: 0.005) 参数tau定义了目标网络软更新时的平滑系数，
                     # 它控制了新目标值在更新目标网络时与旧目标值的融合程度。
                     # 较小的tau值会导致目标网络变化较慢，从而增加了训练的稳定性，但也可能降低学习速度。
        'lr':0.0003, # learning rate (default: 0.0003)
        'alpha':0.2, # Temperature parameter α determines the relative importance of the entropy\term against the reward (default: 0.2)
        'automatic_entropy_tuning':False, # Automaically adjust α (default: False)
        'batch_size':512, # batch size (default: 256)
        'num_steps':1000, # maximum number of steps (default: 1000000)
        'hidden_sizes':[1024,512,256], # 隐藏层大小，带有激活函数的隐藏层层数等于这一列表大小
        'updates_per_step':1, # model updates per simulator step (default: 1) 每步对参数更新的次数
        'start_steps':1000, # Steps sampling random actions (default: 10000) 在开始训练之前完全随机地进行动作以收集数据
        'target_update_interval':10, # Value target update per no. of updates per step (default: 1) 目标网络更新的间隔
        'replay_size':10000000, # size of replay buffer (default: 10000000)
        'cuda':True, # run on CUDA (default: False)
        'LOAD PARA': True, #是否读取参数
        'task':'Train', # 测试或训练或画图，Train,Test,Plot
        'activation':nn.ReLU, #激活函数类型
        'plot_type':'2D-2line', #'3D-1line'为三维图，一条曲线；'2D-2line'为二维图，两条曲线
        'plot_title':'reward-steps.png'}  
                    
# Environment
# env = NormalizedActions(gym.make(args.env_name))

sma = 7171e3
thrust = 1e-2
ref_p = 2e3
T_f = 2000
discreT = 50
Cd = np.eye(6)
R = np.eye(3)*1e6
Q11 = np.eye(3)*1e1
Q22 = np.eye(3)*2
S11 = np.eye(3)
S22 = np.eye(3)

env_f = env.Rel_trans(sma, thrust, ref_p, T_f, discreT,Cd,R,Q11,Q22,S11,S22,pre_horizon = 5)
# Agent
agent = SAC(env_f.observation_space.shape[0], env_f.action_space, args) #discrete不能用shape，要用n提取维度数量
#Tensorboard
'''创建一个SummaryWriter对象，用于将训练过程中的日志信息写入到TensorBoard中进行可视化。
   SummaryWriter()这是创建SummaryWriter对象的语句。SummaryWriter是TensorBoard的一个API，用于将日志信息写入到TensorBoard中。
   format括号里内容是一个字符串格式化的表达式，用于生成一个唯一的日志目录路径。{}是占位符，format()方法会将占位符替换为对应的值
   datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")：这是获取当前日期和时间，并将其格式化为字符串。
   strftime()方法用于将日期时间对象转换为指定格式的字符串。在这里，日期时间格式为"%Y-%m-%d_%H-%M-%S"，表示年-月-日_小时-分钟-秒。
   "autotune" if args.automatic_entropy_tuning else ""：这是一个条件表达式，用于根据args.automatic_entropy_tuning的值来决定是否插入"autotune"到日志目录路径中。
   'runs/{}_SAC_{}_{}_{}'是一个字符串模板，其中包含了四个占位符 {}。当使用 format() 方法时，传入的参数会按顺序替换占位符，生成一个新的字符串。'''
#writer = SummaryWriter('runs/{}_SAC_{}_{}.txt'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),args['policy'], "autotune" if args['automatic_entropy_tuning'] else ""))

# Memory
memory = ReplayMemory(args['replay_size'])

if args['task']=='Train':
    # Training Loop
    updates = 0
    best_avg_reward=0
    total_numsteps=0
    steps_list=[]
    episode_reward_list=[]
    avg_reward_list=[]
    if args['LOAD PARA']==True:
        agent.load_checkpoint("sofarsogood.pt")
        best_avg_reward = 40
    for i_episode in itertools.count(1): #itertools.count(1)用于创建一个无限迭代器。它会生成一个连续的整数序列，从1开始，每次递增1。
        success=False
        episode_reward = 0
        done=False
        episode_steps = 0
        state = env_f.reset()
        while True:
            # if args['start_steps'] > total_numsteps:
            #     action = env.attack_action_space.sample()   # 开始训练前随机动作若干次获取数据
            # else:
            action = agent.select_action(state)  # 开始输出actor网络动作
            if len(memory) > args['batch_size']:
                # Number of updates per step in environment 每次交互之后可以进行多次训练...
                for i in range(args['updates_per_step']):
                    # Update parameters of all the networks
                    # print("i",i)
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args['batch_size'], updates)
                    #print(critic_1_loss)
                    updates += 1


            next_state, reward, done, info= env_f.step(action) # Step
            episode_steps += 1
            episode_reward += reward #没用gamma是因为在sac里求q的时候用了
            total_numsteps+=1
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            memory.push(state, action, reward, next_state, done) # Append transition to memory
            state = next_state

            if done:
                #env.plot(i_episode,episode_steps)
                steps_list.append(i_episode)
                episode_reward_list.append(episode_reward)
                if len(episode_reward_list)>=500:
                    avg_reward_list.append(sum(episode_reward_list[-500:])/500)
                else:
                    avg_reward_list.append(sum(episode_reward_list)/len(episode_reward_list))
                if info:
                    success=True
                break
        #writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, episode steps: {}, reward: {}, success: {}".format(i_episode, episode_steps, round(episode_reward, 4), success))
        # round(episode_reward,2) 对episode_reward进行四舍五入，并保留两位小数

        if i_episode % 100 == 0 and args['eval'] is True: #评价上一个训练过程
            avg_reward = 0.
            episodes = 10
            if args['LOAD PARA']==True:
                episodes = 40
            done_num=0
            for _  in range(episodes):
                state = env_f.reset()
                episode_reward = 0
                done = False
                steps=0
                while not done:
                    action = agent.select_action(state, evaluate=False) #evaluate为True时为确定性网络，直接输出mean
                    next_state, reward, done, info = env_f.step(action)
                    episode_reward += reward
                    state = next_state
                    steps+=1
                    if info:
                        done_num+=1
                    if done:
                        print(steps)
                        break
                avg_reward += episode_reward
            avg_reward /= episodes
            #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {},完成数：{}".format(episodes, round(avg_reward, 4),done_num))
            print("----------------------------------------")
            if avg_reward>best_avg_reward:
                best_avg_reward=avg_reward
                agent.save_checkpoint('sofarsogood.pt')
            if done_num==episodes and avg_reward_list[-1]>300:
                agent.save_checkpoint("approach-no-impulse.pt")
                env_f.plot(args, steps_list, episode_reward_list, avg_reward_list)
                break
        if i_episode==99999:
            agent.save_checkpoint("approach-no-impulse.pt")
            env_f.plot(args, steps_list, episode_reward_list, avg_reward_list)
            break

if args['task']=='Test':
    agent.load_checkpoint("approach-no-impulse.pt")
    avg_reward = 0
    episodes = 10
    done_num=0
    for i  in range(episodes):
        state = env_f.reset()
        episode_reward = 0
        done = False
        steps=0
        while not done:
            action = agent.select_action(state, evaluate=False) #evaluate为True时为确定性网络，直接输出mean
            next_state, reward, done, info = env_f.step(action)
            episode_reward += reward
            state = next_state
            steps+=1
            if info:
                done_num+=1
            if done:
                break
        avg_reward += episode_reward
        print("EP_reward:",episode_reward)
    avg_reward /= episodes
    #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {},完成数：{}".format(episodes, round(avg_reward, 4),done_num))
    print("----------------------------------------")