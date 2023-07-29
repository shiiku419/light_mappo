import gym.spaces
import numpy as np
import random
import math
from scipy.special import softmax
from envs.ga import genetic_algorithm
from torch.utils.tensorboard import SummaryWriter
from pymcdm.correlations import pearson, weighted_spearman
from pymcdm.methods import PROMETHEE_II
from pymcdm.helpers import rrankdata
import torch
import csv

f = open('action.csv', 'w')
f2 = open('generate.csv', 'w')
f3 = open('dataset.csv', 'w')
writer = csv.writer(f)
writer2 = csv.writer(f2)
writer3 = csv.writer(f3)

class EnvCore(object):
    
    def __init__(self, n_member=5):
        self.agent_num = 5
        self.obs_dim = 14*5
        self.action_dim = 7
        self.dataset = np.random.rand(7, 7)
        self.writer = SummaryWriter(log_dir='runs/experiment_name')
        self.n_member = n_member
        self.n_action = 7
        self.action_space = gym.spaces.Dict(
            {
                i: gym.spaces.Dict(
                    {
                        "p_thresholds": gym.spaces.Box(
                            low=0, high=1, shape=(7,), dtype=float
                        ),
                        "n_thresholds": gym.spaces.Box(
                            low=0, high=1, shape=(7,), dtype=float
                        ),
                        "matrix": gym.spaces.Box(
                            low=0, high=1, shape=(7,), dtype=float
                        ),
                    }
                )
                for i in range(self.n_member)
            }
        )


        self.observation_space = gym.spaces.Dict({
            i: gym.spaces.Dict({
                'ranking_difference': gym.spaces.Box(low=-10, high=10, shape=(self.n_member, 7, 2), dtype=float),
                'thresholds': gym.spaces.Dict({
                    'p_thresholds': gym.spaces.Box(low=0, high=1, shape=(7,), dtype=float),
                    'n_thresholds': gym.spaces.Box(low=0, high=1, shape=(7,), dtype=float)
                })
            }) for i in range(self.n_member)
        })

        self.log = []
        self.episode = 1
        self.max_step = 100
        self.agent = random.sample(range(self.n_member), self.n_member)
        self.prom = PROMETHEE_II(preference_function='usual')

        self.criterion_type = self.set_criterion()
        #self.WP, self.w = self.idocriw_method(self.dataset, self.criterion_type)

        self.W = None
        self.P = {}
        self.Q = {}
        self.F = ['t6' for _ in range(7)]

        self.pre_threshold = [[0 for _ in range(7)] for _ in range(self.n_member)]
        self.first_ranking = self.get_ranking(self.F, self.dataset, self.criterion_type)

        self.ranking = self.first_ranking.copy()

        self.reward_count = [0 for _ in range(self.n_member)]
        self.step_count = 0
        self.penalty=[0 for _ in range(self.n_member)]
        self.time = 0

        self.params = {}

    def step(self, actions):
        self.time += 1
        self.generator = random.sample(range(self.n_member), self.n_member)

        action = {}
        rewards = []
        post_psis = []
        done = False
        for agent_id in self.agent:
            default_action = actions[agent_id]
            split_actions = np.array_split(default_action, 3)
            action = split_actions[0].flatten()
            subaction = split_actions[1].flatten()
            subsubaction = split_actions[2].flatten()
            self.ranking, penalty = self.change_ranking(
                action, subaction, agent_id, self.dataset, self.ranking
            )
            _observation, observation = self.get_observation(self.ranking)

            reward, post_psi, params = self.get_reward(penalty, agent_id)
            self.reward_count[agent_id] += reward
            self.penalty[agent_id] += penalty
            
            rewards.append(reward)
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
                    
        done = self.check_is_done(post_psis)

        if self.time% 25 == 0 and self.generator[0] == 0:
            self.generate(subsubaction)

        if self.time == 1:
            self.log = []
            #self.log.append(self.dataset)
        if done:
            writer.writerow([self.episode, agent_id, '+', self.P[agent_id]])
            writer.writerow([self.episode, agent_id, '-', self.Q[agent_id]])
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
            #self.log_reshaped = self.log.view(-1, self.log.shape[-1])  # reshape into 2D
            self.reward_count = [0 for _ in range(self.n_member)]
            self.penalty=[0 for _ in range(self.n_member)]
            self.episode += 1
        return observation, rewards, done, info

    def generate(self, subaction):
        random = np.random.randint(0, 4)
        index = np.where(self.ranking[random] == self.ranking[random].max(0)[1])[0][0]
        self.dataset = np.insert(self.dataset, index, subaction.tolist(), axis=1)
        print(self.dataset.shape)


    def reset(self):
        self.time = 0
        self.criterion_type = self.set_criterion()
        self.agent = random.sample(range(self.n_member), self.n_member)
        self.dataset = np.random.rand(7, 7)
        writer3.writerow([self.episode, self.dataset])
        self.first_ranking = self.get_ranking(self.F, self.dataset, self.criterion_type)
        _, observation = self.get_observation(self.first_ranking)
        return observation

    def close(self):
        pass

    def seed(self):
        pass

    def set_criterion(self):
        type = ["max", "min"]
        prob = [0.7, 0.3]
        self.criterion_type = np.random.choice(a=type, size=7, p=prob)
        return self.criterion_type

    def get_satisfaction(self, id):
        psi, gsi = self.calc_satisfaction(self.distance, self.first_ranking, 1, 7)

        post_psi, post_gsi = self.calc_satisfaction(self.distance, self.ranking, 1, 7)

        params = {
            "pre_psi": psi[id],
            "post_psi": post_psi[id],
            "pre_gsi": gsi,
            "post_gsi": post_gsi,
        }

        return params, post_psi

    def get_reward(self, penalty, id):
        params, post_psi = self.get_satisfaction(id)

        reward = 0
        clip = 0

        main_reward = params["post_psi"] - max(penalty, 1e-10)
        sub_reward = params["post_gsi"]

        clip = main_reward + (sub_reward / self.n_member*2)

        reward = clip

        return reward, post_psi, params

    def get_observation(self, p):
        group_rank = self.calc_group_rank(p)
        observations = []

        for i in range(len(self.first_ranking)):
            observations.append(self.first_ranking[i])

        return observations, observations

    def check_is_done(self, post_psi):
        if all(0.8 <= flag for flag in post_psi):
            return True
        else:
            return self.time == self.max_step

    def target_function(self, variable):
        epsilon = 1e-8
        variable_sum = sum(variable)

        if variable_sum > epsilon:
            variable = [
                variable[i] / (variable_sum + epsilon) for i in range(0, len(variable))
            ]
        else:
            variable = [variable[i] / sum(variable) for i in range(0, len(variable))]
        WP_s = np.copy(self.WP)
        for i in range(0, self.WP.shape[0]):
            for j in range(0, self.WP.shape[1]):
                WP_s[i, j] = WP_s[i, j] * variable[j]
        total = abs(WP_s.sum(axis=1))
        total = sum(total)
        return total

   
    def solution(self):
        result = [np.random.rand(1, 7)[0] for _ in range(self.n_member)]
        return result
     
    def distance_matrix(self, dataset, criteria=0):
        distance_array = np.zeros(shape=(dataset.shape[0], dataset.shape[0]))
        for i in range(0, distance_array.shape[0]):
            for j in range(0, distance_array.shape[1]):
                distance_array[i, j] = dataset[i, criteria] - dataset[j, criteria]
        return distance_array

    def preference_degree(self, dataset, W, Q, S, P, F):
        pd_array = np.zeros(shape=(dataset.shape[0], dataset.shape[0]))
        for k in range(0, dataset.shape[1]):
            distance_array = self.distance_matrix(dataset, criteria=k)
            for i in range(0, distance_array.shape[0]):
                for j in range(0, distance_array.shape[1]):
                    if i != j:
                        if F[k] == "t1":
                            if distance_array[i, j] <= 0:
                                distance_array[i, j] = 0
                            else:
                                distance_array[i, j] = 1
                        if F[k] == "t2":
                            if distance_array[i, j] <= Q[k]:
                                distance_array[i, j] = 0
                            else:
                                distance_array[i, j] = 1
                        if F[k] == "t3":
                            if distance_array[i, j] <= 0:
                                distance_array[i, j] = 0
                            elif (
                                distance_array[i, j] > 0
                                and distance_array[i, j] <= P[k]
                            ):
                                distance_array[i, j] = distance_array[i, j] / P[k]
                            else:
                                distance_array[i, j] = 1
                        if F[k] == "t4":
                            if distance_array[i, j] <= Q[k]:
                                distance_array[i, j] = 0
                            elif (
                                distance_array[i, j] > Q[k]
                                and distance_array[i, j] <= P[k]
                            ):
                                distance_array[i, j] = 0.5
                            else:
                                distance_array[i, j] = 1
                        if F[k] == "t5":
                            if distance_array[i, j] <= Q[k]:
                                distance_array[i, j] = 0
                            elif (
                                distance_array[i, j] > Q[k]
                                and distance_array[i, j] <= P[k]
                            ):
                                distance_array[i, j] = (distance_array[i, j] - Q[k]) / (
                                    P[k] - Q[k]
                                )
                            else:
                                distance_array[i, j] = 1
                        if F[k] == "t6":
                            if distance_array[i, j] <= 0:
                                distance_array[i, j] = 0
                            else:
                                distance_array[i, j] = 1 - math.exp(
                                    -(distance_array[i, j] ** 2) / (2 * S[k] ** 2)
                                )
                        if F[k] == "t7":
                            if distance_array[i, j] == 0:
                                distance_array[i, j] = 0
                            elif (
                                distance_array[i, j] > 0
                                and distance_array[i, j] <= S[k]
                            ):
                                distance_array[i, j] = (
                                    distance_array[i, j] / S[k]
                                ) ** 0.5
                            elif distance_array[i, j] > S[k]:
                                distance_array[i, j] = 1
            pd_array = pd_array + softmax(W, axis=0)[k] * distance_array
        return pd_array

    def promethee_ii(self, dataset, W, Q, S, P, F, sort=True, topn=0, graph=False):
        pd_matrix = self.preference_degree(dataset, W, Q, S, P, F)
        flow_plus = np.sum(pd_matrix, axis=1) / (pd_matrix.shape[0] - 1)
        flow_minus = np.sum(pd_matrix, axis=0) / (pd_matrix.shape[0] - 1)
        flow = flow_plus - flow_minus
        flow = np.reshape(flow, (pd_matrix.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, pd_matrix.shape[0] + 1)), axis=1)
        if sort == True or graph == True:
            flow = flow[np.argsort(flow[:, 1])]
            flow = flow[::-1]
        if topn > 0:
            if topn > pd_matrix.shape[0]:
                topn = pd_matrix.shape[0]
        return flow

    def distance(self, j, g_rank):
        return abs(j - g_rank) ** 2
    
    def calc_satisfaction(self, func, p, frm, to):
        group_satisfaction = 0
        satisfaction_index = [0 for _ in range(self.n_member)]
        g_ranks = self.calc_group_rank(p)
        
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


    def calc_group_rank(self, p):
        group_rank = np.copy(p[0])
        for i in range(1, len(p)):
            group_rank += p[i]
        group_rank = group_rank / len(p)
        # group_rank = group_rank[np.argsort(group_rank[:, 1])]
        observation = group_rank
        return observation

    def get_ranking(self, F, dataset, criterion_type):
        self.W = self.solution()
        rank = {}

        for k in range(self.n_member):
            i = self.agent[k]
            p = []

            self.P[i] = [random.random() * 10 for _ in range(7)]
            self.Q[i] = [random.uniform(0, self.P[i][j]) for j in range(7)]
            S = [(self.P[i][j] - self.Q[i][j]) if self.P[i][j] != self.Q[i][j] else 1e-10 for j in range(7)]

            self.pre_threshold[k] = S

            pref = self.prom(dataset, self.W[i], self.F, p=self.P[i], q=self.Q[i])
            ranking = rrankdata(pref)
            
            for R, P in zip(ranking, pref):
                p.append([R, P])
                
            p = np.vstack(p)
            rank[i] = p

        return rank

    def change_ranking(self, action, subaction, id, dataset, p):
        self.P[id] = np.clip(self.P[id] + action, 0, 10)
        self.Q[id] = np.clip(self.Q[id] + subaction, 0, 10)
        rank = []

        S = [(self.P[id][j] - self.Q[id][j]) if self.P[id][j] != self.Q[id][j] else 1e-10 for j in range(7)]

        scale = 1/700
        penalty = sum([scale * max(0, S[i] - self.pre_threshold[id][i])**2 for i in range(len(S))])

        pref = self.prom(dataset, self.W[id], self.F, p=self.P[id], q=self.Q[id])
        ranking = rrankdata(pref)
        
        for R, P in zip(ranking, pref):
            rank.append([R, P])

        rank = np.vstack(rank)
        p[id] = rank

        return p, penalty

