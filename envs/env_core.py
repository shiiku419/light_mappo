import gym.spaces
import numpy as np
import random
import math
from scipy.special import softmax
from envs.ga import genetic_algorithm
from torch.utils.tensorboard import SummaryWriter
import torch

class EnvCore(object):
    
    def __init__(self, n_member=5):
        self.agent_num = 5
        self.obs_dim = 5
        self.action_dim = 5
        self.dataset = np.random.rand(5, 5)
        self.writer = SummaryWriter(log_dir='runs/experiment_name')
        self.n_member = n_member
        self.n_action = 5
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
                    }
                )
                for i in range(self.n_member)
            }
        )


        self.observation_space = gym.spaces.Dict({i: gym.spaces.Box(low=0, high=10, shape=(5,), dtype=float) for i in range(self.n_member)})
        self.time = 0
        self.log = []
        self.episode = 0
        self.max_step = 100
        self.agent = random.sample(range(self.n_member), self.n_member)

        self.criterion_type = self.set_criterion()
        self.WP, self.w = self.idocriw_method(self.dataset, self.criterion_type)

        self.W = None
        self.P = {}
        self.Q = {}
        self.F = {}

        self.first_ranking = self.get_ranking(self.F, self.dataset, self.criterion_type)

        self.ranking = self.first_ranking.copy()

        self.pre_threshold = 0
        self.step_count = 0
        self.reward_count = 0

        self.params = {}

    def step(self, actions):
        self.time += 1
        self.step_count += 1
        self.generator = random.sample(range(self.n_member), self.n_member)

        action = {}
        rewards = []
        self.reward_count = [0 for _ in range(self.n_member)]
        observations = {}
        post_psis = {i: 0 for i in range(self.n_member)}
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
            observation = self.get_observation(self.ranking)
            reward, post_psi, params = self.get_reward(penalty, agent_id)
            self.reward_count[agent_id] += reward
            rewards.append(reward)
            post_psis[agent_id] = post_psi
            info = {
                'post_psis': post_psis,
                'time': self.time,
                'reward': self.reward_count,
                'dataset': self.dataset,
                'post_gsi': params['post_gsi']
            }
                    
        done = self.check_is_done(post_psis)

        if self.time% 25 == 0 and self.generator[0] == 0:
            self.generate(subsubaction)

        if self.time == 1:
            self.log = []
            self.log.append(self.dataset)
        if done:
            self.episode += 1
            self.log.append(self.dataset)
            self.log = torch.Tensor(self.log)
            
            for i, post_psi in enumerate(post_psis):
                self.writer.add_scalar('post_psis/agent_{}'.format(i), post_psi, self.step_count)
                self.writer.add_scalar('reward/agent_{}'.format(i), info['reward'][i], self.step_count)
            self.writer.add_scalar('log/time', info['time'], self.step_count)
            self.writer.add_scalar('log/post_gsi', info['post_gsi'], self.step_count)
            self.log_reshaped = self.log.view(-1, self.log.shape[-1])  # reshape into 2D
            self.writer.add_embedding(self.log_reshaped,
                        global_step=self.episode,
                        tag='dataset',
                        metadata=['row_{}'.format(i) for i in range(self.log_reshaped.shape[0])])  # create a label for each row
        return observation, rewards, done, info

    def generate(self, subaction):
        random = np.random.randint(0, 4)
        index = np.where(self.ranking[random] == self.ranking[random].max(0)[1])[0][0]
        self.dataset[index] = subaction.tolist()

    def reset(self):
        self.time = 0
        self.criterion_type = self.set_criterion()
        self.agent = random.sample(range(self.n_member), self.n_member)
        self.dataset = np.random.rand(5, 5)
        self.first_ranking = self.get_ranking(self.F, self.dataset, self.criterion_type)
        observation = self.get_observation(self.first_ranking)
        return observation

    def close(self):
        pass

    def seed(self):
        pass

    def set_criterion(self):
        type = ["max", "min"]
        prob = [0.7, 0.3]
        self.criterion_type = np.random.choice(a=type, size=5, p=prob)
        return self.criterion_type

    def get_satisfaction(self, id):
        psi, gsi = self.calc_satisfaction(self.distance, self.first_ranking, 1, 5)

        post_psi, post_gsi = self.calc_satisfaction(self.distance, self.ranking, 1, 5)

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

        main_reward = params["post_psi"] - params["pre_psi"]
        sub_reward = params["post_gsi"] - params["pre_gsi"]

        clip = main_reward + (sub_reward / self.n_member)
        max_reward = 1
        min_reward = -1

        # 報酬の正規化
        clip = (clip - min_reward) / (max_reward - min_reward)

        reward = clip  # 報酬として満足度の差分そのものを使用

        # もしくは、満足度の絶対値を報酬として使用する場合
        reward = abs(clip) - penalty

        return reward, post_psi, params

    def get_observation(self, p):
        group_rank = self.calc_group_rank(p)
        observation = [
            group_rank[:, 1] - self.first_ranking[i][:, 1] for i in range(self.n_member)
        ]
        return observation

    def check_is_done(self, post_psi):
        if all(0.8 <= flag for flag in post_psi) == True:
            return True
        else:
            return self.time == self.max_step

    def idocriw_method(self, dataset, criterion_type):
        X = np.copy(dataset)
        X = X / X.sum(axis=0)
        X_ln = np.copy(dataset)
        X_r = np.copy(dataset)
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                X_ln[i, j] = X[i, j] * math.log(X[i, j])
        d = np.zeros((1, X.shape[1]))
        self.w = np.zeros((1, X.shape[1]))
        for i in range(0, d.shape[1]):
            d[0, i] = 1 - (-1 / (math.log(d.shape[1])) * sum(X_ln[:, i]))
        for i in range(0, self.w.shape[1]):
            self.w[0, i] = d[0, i] / d.sum(axis=1)
        for i in range(0, len(criterion_type)):
            if criterion_type[i] == "min":
                X_r[:, i] = dataset[:, i].min() / X_r[:, i]
        X_r = X_r / X_r.sum(axis=0)
        a_max = X_r.max(axis=0)
        A = np.zeros(dataset.shape)
        np.fill_diagonal(A, a_max)
        for k in range(0, A.shape[0]):
            i, _ = np.where(X_r == a_max[k])
            i = i[0]
            for j in range(0, A.shape[1]):
                A[k, j] = X_r[i, j]
        a_max_ = A.max(axis=0)
        P = np.copy(A)
        for i in range(0, P.shape[1]):
            P[:, i] = (-P[:, i] + a_max_[i]) / a_max[i]
        self.WP = np.copy(P)
        np.fill_diagonal(self.WP, -P.sum(axis=0))
        return self.WP, self.w

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
        solution = genetic_algorithm(
            population_size=5,
            mutation_rate=0.1,
            elite=1,
            min_values=[0] * self.WP.shape[1],
            max_values=[1] * self.WP.shape[1],
            eta=1,
            mu=1,
            generations=100,
            target_function=self.target_function,
        )
        solution = solution[:-1]
        solution = solution / sum(solution)
        w_ = np.copy(self.w)
        w_ = w_ * solution
        w_ = w_ / w_.sum()
        w_ = w_.T

        result = [item for i in w_ for item in i]

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
        result = 0
        satisfaction = 0
        group_satisfaction = 0
        satisfaction_index = [0 for _ in range(self.n_member)]
        g_ranks = self.calc_group_rank(p)
        g_ranks = g_ranks[np.argsort(g_ranks[:, 1])]
        for k in range(0, len(p)):
            i = self.agent[k]
            i_ranks = p[i][np.argsort(p[1][:, 1])]

            for j in range(frm, to + 1):
                g_rank = np.where(g_ranks == i_ranks[j - 1][0])[0][0] + 1
                result += func(j, g_rank)

            bottom = to**3 - to
            satisfaction = 1 - 6 * result / bottom
            group_satisfaction += satisfaction
            satisfaction_index[i] = satisfaction
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
        self.W = [self.solution() for _ in range(5)]
        pref = ["t1", "t2", "t3", "t4", "t5", "t6"]

        p = {}

        for k in range(self.n_member):
            i = self.agent[k]

            self.P[i] = [random.random() for _ in range(5)]
            self.Q[i] = [random.uniform(0, self.P[i][j]) for j in range(5)]
            S = [(self.P[i][j] + self.Q[i][j] / 2) for j in range(5)]

            F[i] = [pref[random.randint(0, 5)] for _ in range(5)]

            self.pre_threshold = sum(S)

            p[i] = self.promethee_ii(
                dataset,
                W=self.W[i],
                Q=self.Q[i],
                S=S,
                P=self.P[i],
                F=F[i],
                sort=False,
                topn=10,
                graph=False,
            )
        return p

    def change_ranking(self, action, subaction, id, dataset, ranking):
        #self.P[id] = [x + y for (x, y) in zip(self.P[id], action.tolist())]
        self.P[id] += action
        self.Q[id] =+ subaction
        S = [(self.P[id][j] + self.Q[id][j]) / 2 for j in range(5)]

        penalty = sum(S) - self.pre_threshold

        ranking[id] = self.promethee_ii(
            dataset,
            W=self.W[id],
            Q=self.Q[id],
            S=S,
            P=self.P[id],
            F=self.F[id],
            sort=False,
            topn=10,
            graph=False,
        )
        return ranking, penalty
