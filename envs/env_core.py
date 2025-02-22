import gym.spaces
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from pymcdm.correlations import pearson, weighted_spearman
from pymcdm.methods import PROMETHEE_II
from pymcdm.helpers import rrankdata
import torch
import csv
import sys
import os
from pathlib import Path

f = open('action.csv', 'w')
f2 = open('generate.csv', 'w')
f3 = open('dataset.csv', 'w')
writer = csv.writer(f)
writer2 = csv.writer(f2)
writer3 = csv.writer(f3)

run_dir2 = (
    Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/logs")
)
run_dir3 = (
    Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/logs2")
)
if not run_dir2.exists():
    os.makedirs(str(run_dir2))
    os.makedirs(str(run_dir3))

class EnvCore(object):
    
    def __init__(self, n_member=5):
        self.agent_num = 5
        self.obs_dim = 10 * 5
        self.action_dim = 15 + 1
        self.dataset = np.random.rand(5, 5)
        self.writer = SummaryWriter(log_dir='runs/experiment_name')
        self.n_member = n_member
        self.n_action = 4
        self.action_space = gym.spaces.Dict(
            {
                i: gym.spaces.Dict(
                    {
                        "p_thresholds": gym.spaces.Box(
                            low=0, high=1, shape=(5,), dtype=float
                        ),
                        "n_thresholds": gym.spaces.Box(
                            low=0, high=1, shape=(5,), dtype=float
                        ),
                        "matrix": gym.spaces.Box(
                            low=0, high=1, shape=(5,), dtype=float
                        ),
                        "propose": gym.spaces.Discrete(2),
                    }
                )
                for i in range(self.n_member)
            }
        )


        self.observation_space = gym.spaces.Dict({
            i: gym.spaces.Dict({
                'ranking_difference': gym.spaces.Box(low=-10, high=10, shape=(self.n_member, 5, 2), dtype=float),
                'thresholds': gym.spaces.Dict({
                    'p_thresholds': gym.spaces.Box(low=0, high=1, shape=(5,), dtype=float),
                    'n_thresholds': gym.spaces.Box(low=0, high=1, shape=(5,), dtype=float)
                })
            }) for i in range(self.n_member)
        })

        self.log = []
        self.episode = 0
        self.max_step = 100
        self.agent = random.sample(range(self.n_member), self.n_member)
        self.prom = PROMETHEE_II(preference_function='vshape')

        self.criterion_type = self.set_criterion()

        self.W = None
        self.P = {}
        self.Q = {}
        self.F = ['t6' for _ in range(5)]

        self.pre_threshold = [[0 for _ in range(5)] for _ in range(self.n_member)]
        self.pre_index = {
            'p': [[0 for _ in range(5)] for _ in range(self.n_member)],
            'q': [[0 for _ in range(5)] for _ in range(self.n_member)]
        }
        self.first_ranking = self.get_ranking(self.F, self.dataset, self.criterion_type)

        self.ranking = self.first_ranking.copy()

        self.reward_count = [0 for _ in range(self.n_member)]
        self.step_count = 0
        self.penalty=[0 for _ in range(self.n_member)]
        self.time = 0

        self.params = {}
        self.rewards = [0 for _ in range(self.n_member)]
        self.logger = self.create_new_writer(self.episode)
        self.logger2 = self.create_new_writer2(self.episode)
        
    def create_new_writer(self, episode):
        self.f = open(f'logs/action_{episode}.csv', 'w')
        return csv.writer(self.f)
    
    def create_new_writer2(self, episode):
        self.f2 = open(f'logs2/action_{episode}.csv', 'w')
        return csv.writer(self.f2)

    def step(self, actions):
        self.time += 1
        action = {}
        post_psis = []
        done = False
        for agent_id in self.agent:
            default_action = actions[agent_id]
            split_actions = np.array_split(default_action[:-1], 3)
            action = split_actions[0].flatten()
            subaction = split_actions[1].flatten()
            subsubaction = split_actions[2].flatten()
            propose = default_action[-1]
            
            if propose == 1:
                self.generate(subsubaction)
                self.logger2.writerow([self.step_count, agent_id, subsubaction])
                
            self.ranking = self.change_ranking(
                action, subaction, agent_id, self.dataset, self.ranking
            )
            
            self.ranking = self.update_ranking(self.F, self.dataset, self.criterion_type)
            _observation, observation = self.get_observation(self.ranking)
            
            self.logger.writerow([self.step_count, agent_id, '+', self.P[agent_id]])
            self.logger.writerow([self.step_count, agent_id, '-', self.Q[agent_id]])

            reward, post_psi, params, penalty = self.get_reward(agent_id, propose)
            self.reward_count[agent_id] += reward
            self.penalty[agent_id] += penalty
            
            self.rewards[agent_id] = reward
            post_psis = post_psi
            info = {
                'post_psis': post_psis,
                'time': self.time,
                'reward': self.reward_count,
                'dataset': self.dataset,
                'post_gsi': params['post_gsi'],
                'penalty': self.penalty
            }
            self.step_count += 1

        done = self.check_is_done(post_psi)

        if done:
            writer2.writerow([self.episode, agent_id, subsubaction])
            writer3.writerow([self.episode, self.dataset])
            #self.log.append(self.dataset)
            self.log = torch.Tensor(np.array(self.log))
            
            for i in range(self.n_member):
                self.writer.add_scalar('post_psis/agent_{}'.format(i), info['post_psis'][i], self.step_count)
                self.writer.add_scalar('reward/agent_{}'.format(i), info['reward'][i], self.step_count)
                self.writer.add_scalar('penalty/agent_{}'.format(i), info['penalty'][i], self.step_count)
            self.writer.add_scalar('log/time', info['time'], self.step_count)
            self.writer.add_scalar('log/post_gsi', info['post_gsi'], self.step_count)
            self.writer.add_scalar('log/dataset', self.dataset.shape[0], self.step_count)
            #self.log_reshaped = self.log.view(-1, self.log.shape[-1])  # reshape into 2D
            self.reward_count = [0 for _ in range(self.n_member)]
            self.penalty = [0 for _ in range(self.n_member)]
            self.f.close()
            self.f2.close()

        return [observation, self.rewards, done, info]

    def update_ranking(self, F, dataset, criterion_type):
        rank = {}

        for k in range(self.n_member):
            i = self.agent[k]
            p = []

            pref = self.prom(dataset, self.W[i], self.F, p=self.P[i], q=self.Q[i])
            ranking = rrankdata(pref)
            
            for R, P in zip(ranking, pref):
                p.append([R, P])
                
            p = np.vstack(p)
            rank[i] = p

        return rank

    def generate(self, subaction):
        self.dataset = np.insert(self.dataset, 0, subaction.tolist(), axis=0)

    def reset(self):
        self.episode += 1
        self.rewards = [0 for _ in range(self.n_member)]
        self.logger = self.create_new_writer(self.episode)
        self.logger2 = self.create_new_writer2(self.episode)
        self.time = 0
        self.criterion_type = self.set_criterion()
        self.agent = random.sample(range(self.n_member), self.n_member)
        self.dataset = np.random.rand(5, 5)
        self.logger2.writerow([self.step_count, self.dataset])
        writer3.writerow([self.episode, self.dataset])
        self.first_ranking = self.get_ranking(self.F, self.dataset, self.criterion_type)
        _, observation = self.get_observation(self.first_ranking)
        for agent_id in range(self.n_member):    
            self.logger.writerow([self.step_count, agent_id, '+', self.P[agent_id]])
            self.logger.writerow([self.step_count, agent_id, '-', self.Q[agent_id]])
        #TODO add thresholds
        return observation

    def close(self):
        pass

    def seed(self):
        pass

    def check_is_done(self, post_psi):
        if all(psi >= 0.8 for psi in post_psi):
            return True
        return self.time == self.max_step

    def set_criterion(self):
        type = ["max", "min"]
        prob = [0.7, 0.3]
        self.criterion_type = np.random.choice(a=type, size=5, p=prob)
        return self.criterion_type

    def get_satisfaction(self, id):
        psi, gsi = self.calc_satisfaction(self.distance, self.first_ranking, 1, 5, id)

        post_psi, post_gsi = self.calc_satisfaction(self.distance, self.ranking, 1, 5, id)

        params = {
            "pre_psi": psi[id],
            "post_psi": post_psi[id],
            "pre_gsi": gsi,
            "post_gsi": post_gsi,
        }

        return params, post_psi

    def get_reward(self, id, propose):
        params, post_psi = self.get_satisfaction(id)

        # Threshold-based reward
        penalty_coeff = 1/250
        threshold_change_penalty = penalty_coeff * sum([abs(self.P[id][i] - self.pre_index['p'][id][i])**2 for i in range(len(self.P[id]))])
        threshold_change_penalty += penalty_coeff * sum([abs(self.Q[id][i] - self.pre_index['q'][id][i])**2 for i in range(len(self.Q[id]))])

        # Calculate the final reward
        reward = params["post_psi"] - threshold_change_penalty
        
        if propose == 1:
            reward -= 0.5 

        return reward, post_psi, params, threshold_change_penalty


    def get_observation(self, p):
        observations = []

        for i in range(len(self.first_ranking)):
            observations.append(self.first_ranking[i])

        return observations, observations

    def distance(self, j, g_rank):
        return abs(j - g_rank) ** 2
    
    def calc_satisfaction(self, func, p, frm, to, id):
        group_satisfaction = 0
        satisfaction_index = [0 for _ in range(self.n_member)]
        g_ranks = self.calc_group_rank(p, id)
        
        for k in range(self.n_member):
            i = self.agent[k]
            i_ranks = p[i]
            w_i_rank = p[i][np.argsort(p[1][:, 1])]

            if np.var(i_ranks[:, 1]) == 0 or np.var(g_ranks[:, 1]) == 0:
                satisfaction = 1
            else:
                p_satisfaction = (pearson(i_ranks[:, 1], g_ranks[:, 1])+1)/2
                w_g_rank = g_ranks[np.argsort(g_ranks[:, 1])]
                w_satisfaction = (weighted_spearman(np.flipud(w_i_rank), np.flipud(w_g_rank))+1)/2

                satisfaction =  2 * (p_satisfaction * w_satisfaction) / (p_satisfaction + w_satisfaction)
            
            satisfaction_index[i] = satisfaction
            group_satisfaction += satisfaction_index[i]
        return satisfaction_index, group_satisfaction


    def calc_group_rank(self, p, exclude_id):
        # Sum up all the rankings except for the excluded member.
        group_rank = np.zeros_like(p[0])
        for i in range(len(p)):
            if i != exclude_id:
                group_rank += p[i][np.argsort(p[1][:, 1])]

        # Compute the average ranking (excluding the member).
        group_rank = group_rank / (len(p) - 1)

        return group_rank

    def get_ranking(self, F, dataset, criterion_type):
        self.W = [np.random.rand(1, 5)[0] for _ in range(self.n_member)]
        rank = {}

        for k in range(self.n_member):
            i = self.agent[k]
            p = []

            self.P[i] = [random.random() * 10 for _ in range(5)]
            self.Q[i] = [random.uniform(0, self.P[i][j]) for j in range(5)]
            S = [(self.P[i][j] - self.Q[i][j]) if self.P[i][j] != self.Q[i][j] else 1e-10 for j in range(5)]
            
            self.pre_index['p'][k] = self.P[i]
            self.pre_index['q'][k] = self.Q[i]
            self.pre_threshold[k] = S

            pref = self.prom(dataset, self.W[i], self.F, p=self.P[i], q=self.Q[i])
            ranking = rrankdata(pref)
            
            for R, P in zip(ranking, pref):
                p.append([R, P])
                
            p = np.vstack(p)
            rank[i] = p

        return rank

    def change_ranking(self, action, subaction, id, dataset, p):
        p_delta = self.P[id] + action  # Ensure p_delta is positive
        n_delta = self.Q[id] - subaction  # Ensure n_delta is positive

        self.P[id] = np.clip(p_delta, 0, 10)
        self.Q[id] = np.clip(n_delta, 0, 10)
        rank = []

        pref = self.prom(dataset, self.W[id], self.F, p=self.P[id], q=self.Q[id])
        ranking = rrankdata(pref)

        for R, P in zip(ranking, pref):
            rank.append([R, P])

        rank = np.vstack(rank)
        p[id] = rank

        return p