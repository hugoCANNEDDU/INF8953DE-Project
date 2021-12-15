import numpy as np
from progress.bar import Bar


class Bandit:
  def __init__(self, mu_star=0.9, delta_mu=0.1, best_action=None, nb_action=2):
    self.rng = np.random.default_rng(np.random.randint(0, 100000000))
    self.nb_action = nb_action
    if nb_action == 2:
      self.delta_mu = delta_mu
      if best_action is None:
        self.best_action = self.rng.integers(0, 2)
      else :
        self.best_action = best_action
      self.proba = [0, 0]
      self.proba[self.best_action] = mu_star
      self.proba[int(not self.best_action)] = mu_star - delta_mu
    else:
      self.proba = self.rng.uniform(0.1, 1, nb_action)
      self.best_action = np.argmax(self.proba)

  def pull(self, action):
    return self.rng.binomial(1, self.proba[action])

  def get_regret(self, action):
    regret = abs(self.proba[self.best_action] - self.proba[action])
    return regret


class UCB:
  def __init__(self, bandit, c=2, q_val=0.):
    self.nb_action = bandit.nb_action
    self.q_val = q_val
    self.c = c
    self.bandit = bandit

  def run(self, nb_pull):
    regrets = []
    actions = []

    n = np.zeros(self.nb_action, dtype=np.int32)
    avg_rewards = np.zeros(self.nb_action)

    for i in range(nb_pull):
      if 0 in n:
        action = np.where(n == 0)[0][0]
      else:
        action = np.argmax(avg_rewards + np.sqrt(self.c * np.log(i) / n))
      actions.append(action)

      reward = self.bandit.pull(action)
      n[action] += 1
      avg_rewards[action] += (reward - avg_rewards[action]) / n[action]

      regret = self.bandit.get_regret(action)
      if i == 0:
        regrets.append(regret)
      else:
        regrets.append(regrets[-1] + regret)

    return actions, regrets


class TargetUCB:
  def __init__(self, bandit, nb_neighbor=1, c=2, q_val=0.):
    self.nb_neighbor = nb_neighbor
    self.bandit = bandit
    self.c = c
    self.q_val = q_val

    self.nb_action = bandit.nb_action
    self.n_action = np.zeros(self.nb_action, dtype=np.int32)
    self.t_action = np.zeros(self.nb_action)
    self.avg_rewards = np.zeros(self.nb_action)
    self.regrets = []
    self.t = 0

  def run(self, target_actions, nb_pull):

    for i in range(nb_pull):
      self.next_action([target_actions[i]])

    return self.regrets

  def next_action(self, target_actions):
    self.t += 1
    if self.t > 1:
      for action in target_actions:
        self.t_action[action] += 1 / self.nb_neighbor

    if 0 in self.n_action:
      action = np.where(self.n_action == 0)[0][0]
    else:
      est_opt = np.sqrt(self.c * np.log(self.t) / self.n_action)

      with np.errstate(divide='ignore'):
        target_opt = (self.t_action - self.n_action) / self.t_action
      target_opt[target_opt < 0] = 0
      target_opt = np.sqrt(target_opt)

      Q = self.avg_rewards + est_opt * target_opt

      if np.all(Q == Q[0]):
        action = np.random.choice(range(self.nb_action))
      else:
        action = np.argmax(Q)

    reward = self.bandit.pull(action)
    self.n_action[action] += 1
    self.avg_rewards[action] += (reward - self.avg_rewards[action]) / self.n_action[action]

    regret = self.bandit.get_regret(action)

    if self.t == 1:
      self.regrets.append(regret)
    else:
      self.regrets.append(self.regrets[-1] + regret)

    return action


class Greedy():
  def __init__(self, bandit):
    self.nb_action = 2
    self.bandit = bandit

  def run(self, target_actions, nb_pull):
    regrets = []

    for i in range(nb_pull):

      unique, count = np.unique(target_actions[:i + 1], return_counts=True)
      actual_target_actions = np.zeros(self.nb_action, dtype=np.int32)
      actual_target_actions[unique] = count

      action = np.argmax(actual_target_actions)

      reward = self.bandit.pull(action)
      regret = self.bandit.get_regret(action)

      if i == 0:
        regrets.append(regret)
      else:
        regrets.append(regrets[-1] + regret)

    return regrets


def compute_regret(bandit, results):
  regrets = []
  for result in results:
    regret = 0 if result == "Win" else bandit.delta_mu
    if len(regrets) == 0:
      regrets.append(regret)
    else:
      regrets.append(regrets[-1] + regret)
  return regrets


def compute_result(neighborhood, nb_run, nb_episode, bandits):
  nb_total_neighbor = len(neighborhood)
  target_ucb_regrets_cumul = [[] for _ in range(nb_total_neighbor)]

  for run in range(nb_run):
    bandit = bandits[run]
    if run % 100 == 0:
      print(run)

    agents = []
    nb_neighbor = np.sum(neighborhood, axis=0)
    for i in range(nb_total_neighbor):
      agents.append(TargetUCB(bandit, nb_neighbor[i]))

    actions = np.zeros(nb_total_neighbor, dtype=np.int32)
    for t in range(nb_episode):
      prev_act = actions.copy()

      for i in range(len(agents)):
        neighbor_actions = prev_act[neighborhood[i]]
        actions[i] = agents[i].next_action(neighbor_actions)

    for i, agent in enumerate(agents):
      target_ucb_regrets_cumul[i].append(agent.regrets.copy())

  target_ucb_avg = [np.average(data, axis=0) for data in target_ucb_regrets_cumul]
  target_ucb_full_avg = np.average(target_ucb_avg, axis=0)

  return target_ucb_avg, target_ucb_full_avg
