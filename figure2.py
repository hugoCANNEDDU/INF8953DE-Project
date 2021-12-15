import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import utils

np.random.seed(42)

mu_star = 0.9
nb_episode = 15000
nb_run = 100
delta_mu = 0.1

bandit = utils.Bandit(mu_star, delta_mu)

class Alpha_Optimal():
  def __init__(self, alpha, bandit):
    self.rng = np.random.default_rng(np.random.randint(0, 100000000))
    self.nb_action = 2
    self.alpha = alpha
    self.bandit = bandit

  def run(self, nb_pull):
    rand_pull = self.rng.random(nb_pull)

    actions = np.full(nb_pull, int(not self.bandit.best_action))
    actions[rand_pull < self.alpha] = self.bandit.best_action

    regrets = np.zeros(nb_pull)
    regret = self.bandit.get_regret(int(not self.bandit.best_action))
    regrets[actions != self.bandit.best_action] = regret
    regrets = np.cumsum(regrets)

    return actions, regrets

def compute_result(mu_star, delta_mu, alpha, nb_run, nb_episode):
  bandits = [utils.Bandit(mu_star, delta_mu) for _ in range(nb_run)]

  target_regrets = []
  target_ucb_regrets = []
  greedy_regrets = []
  for i in range(nb_run):

    print(i)

    alpha_optimal = Alpha_Optimal(alpha, bandits[i])
    target_ucb = utils.TargetUCB(bandits[i])
    greedy = utils.Greedy(bandits[i])

    actions, target_regret = alpha_optimal.run(nb_episode)
    target_regrets.append(target_regret)

    target_ucb_regret = target_ucb.run(actions, nb_episode)
    target_ucb_regrets.append(target_ucb_regret)

    greedy_regret = greedy.run(actions, nb_episode)
    greedy_regrets.append(greedy_regret)

  avg_target_regrets = np.average(target_regrets, axis=0)
  avg_target_ucb_regrets = np.average(target_ucb_regrets, axis=0)
  avg_greedy_regrets = np.average(greedy_regrets, axis=0)
  return avg_target_regrets, avg_target_ucb_regrets, avg_greedy_regrets

alpha_optimal_avg_001, target_ucb_avg_001, greedy_avg_001 = compute_result(mu_star, delta_mu, 0.001, nb_run, nb_episode)
alpha_optimal_avg_05, target_ucb_avg_05, greedy_avg_05 = compute_result(mu_star, delta_mu, 0.5, nb_run, nb_episode)
alpha_optimal_avg_09, target_ucb_avg_09, greedy_avg_09 = compute_result(mu_star, delta_mu, 0.9, nb_run, nb_episode)

colors = {"tab:blue": 0.001, "tab:orange": 0.5, "tab:green": 0.9}
line_styles = {"solid": "Target", "dotted": "Target-UCB", "dashed": "Greedy"}

fig, axes = plt.subplots()

axes.plot(alpha_optimal_avg_001, c=list(colors)[0], ls=list(line_styles)[0])
axes.plot(target_ucb_avg_001, c=list(colors)[0], ls=list(line_styles)[1])
axes.plot(greedy_avg_001, c=list(colors)[0], ls=list(line_styles)[2])

axes.plot(alpha_optimal_avg_05, c=list(colors)[1], ls=list(line_styles)[0])
axes.plot(target_ucb_avg_05, c=list(colors)[1], ls=list(line_styles)[1])
axes.plot(greedy_avg_05, c=list(colors)[1], ls=list(line_styles)[2])

axes.plot(alpha_optimal_avg_09, c=list(colors)[2], ls=list(line_styles)[0])
axes.plot(target_ucb_avg_09, c=list(colors)[2], ls=list(line_styles)[1])
axes.plot(greedy_avg_09, c=list(colors)[2], ls=list(line_styles)[2])

patches = []
for key, value in colors.items():
  patches.append(mlines.Line2D([], [], c=key, ls="solid", label=f"α = {value}"))

lines = []
for key, value in line_styles.items():
  lines.append(mlines.Line2D([], [], c="black", ls=key, label=value))

legend1 = plt.legend(handles=patches, loc=2)
legend2 = plt.legend(handles=lines, loc=2, bbox_to_anchor=(0, 0.8))
axes.add_artist(legend1)
axes.add_artist(legend2)
plt.ylabel("Cumulative regret")
plt.xlabel("Episodes")
plt.show()