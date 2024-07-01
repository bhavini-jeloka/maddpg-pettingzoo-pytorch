import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PettingZoo.pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3

from MADDPG import MADDPG


def get_env(env_name, ep_len=25):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=1, num_obstacles=0, max_cycles=ep_len, continuous_actions=False)

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='simple_adversary_v3', help='name of the env',
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3'])
    parser.add_argument('--episode_num', type=int, default=30000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=100,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=5e4,
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    args = parser.parse_args()

    # create folder to save result
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info = get_env(args.env_name, args.episode_length)
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                    result_dir)
    
    eval_ep = []
    evaluation_score_adversary = []
    evaluation_score_agent = []

    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}

    def get_fixed_adversary_policy(env, obs):
        return env.action_space('adversary_0').sample()
        

    def get_fixed_agent_policy(env, obs):
        return env.action_space('agent_0').sample()


    print('Begin Training')

    for episode in range(args.episode_num):
        obs, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < args.random_steps:
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                action = maddpg.select_action(obs)

            next_obs, reward, terminations, done, info = env.step(action)
            # env.render()
            maddpg.add(obs, action, reward, next_obs, done)

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            print(message)

        if args.env_name == 'simple_tag_v3':
            if episode == 0 or (episode + 1) % 500 == 0:  # evaluate every 500 episodes
                eval_ep.append(episode)
                
                print('Begin Evaluation')

                episode_rewards_eval_maddpg = {}

                for agent, ep in episode_rewards.items():
                    section = ep[0:episode]  # Extract the section
                    episode_rewards_eval_maddpg[agent] = section  # Store the section in a new dictionary

                maddpg.save(episode_rewards_eval_maddpg)  # save model

                episode_num_eval = 100 # 1000
                env_eval, dim_info_eval = get_env(args.env_name, args.episode_length)

                folder_name = total_files + 1
                
                maddpg_eval = MADDPG.load(dim_info_eval, os.path.join(result_dir, 'model.pt'))
                episode_rewards_eval_fixed_adversary = {agent_id: np.zeros(episode_num_eval) for agent_id in env_eval.agents}
                
                for episode_eval in range(episode_num_eval):
                    obs_eval, info_eval = env_eval.reset()
                    agent_reward_eval = {agent_id: 0 for agent_id in env_eval.agents}  # agent reward of the current episode

                    for t_eval in range(args.episode_length):

                        action_eval = maddpg_eval.select_action(obs_eval)
                        
                        for agent in action_eval:
                            if agent.find('adversary') != -1: 
                                action_eval[agent] = get_fixed_adversary_policy(env_eval, obs_eval) 

                        next_obs_eval, reward_eval, terminations_eval, done_eval, info_eval = env_eval.step(action_eval)

                        for agent_id, r in reward_eval.items():  # update reward
                            agent_reward_eval[agent_id] += r
                            
                        obs_eval = next_obs_eval

                    env_eval.close()
                    # episode finishes, record reward
                    for agent_id, reward in agent_reward_eval.items():
                        episode_rewards_eval_fixed_adversary[agent_id][episode_eval] = reward
                
                print('Fixed Policy Adversary Over')

                # Evaluation against fixed policy agent
                env_eval, dim_info_eval = get_env(args.env_name, args.episode_length)
                episode_rewards_eval_fixed_agent = {agent_id: np.zeros(episode_num_eval) for agent_id in env_eval.agents}

                for episode_eval in range(episode_num_eval):
                    obs_eval, info_eval = env_eval.reset()
                    agent_reward_eval = {agent_id: 0 for agent_id in env_eval.agents}  # agent reward of the current episode

                    for t_eval in range(args.episode_length):

                        action_eval = maddpg_eval.select_action(obs_eval)
                        
                        for agent in action_eval:
                            if agent.find('agent') != -1: 
                                action_eval[agent] = get_fixed_agent_policy(env_eval, obs_eval) 

                        next_obs_eval, reward_eval, terminations_eval, done_eval, info_eval = env_eval.step(action_eval)

                        for agent_id, r in reward_eval.items():  # update reward
                            agent_reward_eval[agent_id] += r
                            
                        obs_eval = next_obs_eval

                    env_eval.close()
                    # episode finishes, record reward
                    for agent_id, reward in agent_reward_eval.items():
                        episode_rewards_eval_fixed_agent[agent_id][episode_eval] = reward
                
                print('Fixed Policy Agent Over')
                # print average reward for 1000 episodes, don't plot
                evaluation_score_agent.append(np.mean(episode_rewards_eval_fixed_adversary['agent_0']))
                evaluation_score_adversary.append(np.mean(episode_rewards_eval_fixed_agent['adversary_0']))
            
                print('Evaluation Score Agent:', evaluation_score_agent)
                print('Evaluation Score Adversary:', evaluation_score_adversary)
                os.remove(os.path.join(result_dir, 'model.pt'))

                print('End Evaluation')
            
    maddpg.save(episode_rewards)  # save final model   
    print('End Training')


    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward


    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, get_running_reward(reward), label=agent_id)

    if args.env_name == 'simple_tag_v3':
        ax.plot(eval_ep, evaluation_score_agent, c='cyan', label='Evaluation against fixed adversary policy')
        ax.plot(eval_ep, evaluation_score_adversary, c='pink', label='Evaluation against fixed agent policy')
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {args.env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))
