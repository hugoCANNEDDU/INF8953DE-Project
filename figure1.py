import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import utils

np.random.seed(42)

mu_star = 0.9  # @param
nb_episode = 1000  # @param
nb_run = 200  # @param

def compute_result(mu_star, delta_mu, nb_run, nb_episode):
  bandits = [utils.Bandit(mu_star, delta_mu) for _ in range(nb_run)]

  target_regrets = []
  target_ucb_regrets = []
  greedy_regrets = []
  for i in range(nb_run):
    if i % 100 == 0:
      print(i)

    ucb = utils.UCB(bandits[i])
    target_ucb = utils.TargetUCB(bandits[i])
    greedy = utils.Greedy(bandits[i])

    actions, target_regret = ucb.run(nb_episode)
    target_regrets.append(target_regret)

    target_ucb_regret = target_ucb.run(actions, nb_episode)
    target_ucb_regrets.append(target_ucb_regret)

    greedy_regret = greedy.run(actions, nb_episode)
    greedy_regrets.append(greedy_regret)

  avg_target_regrets = np.average(target_regrets, axis=0)
  avg_target_ucb_regrets = np.average(target_ucb_regrets, axis=0)
  avg_greedy_regrets = np.average(greedy_regrets, axis=0)
  return avg_target_regrets, avg_target_ucb_regrets, avg_greedy_regrets


ucb_avg_01, target_ucb_avg_01, greedy_avg_01 = compute_result(mu_star, 0.1, nb_run, nb_episode)
ucb_avg_04, target_ucb_avg_04, greedy_avg_04 = compute_result(mu_star, 0.4, nb_run, nb_episode)
ucb_avg_08, target_ucb_avg_08, greedy_avg_08 = compute_result(mu_star, 0.8, nb_run, nb_episode)

colors = {"tab:blue": 0.1, "tab:orange": 0.4, "tab:green": 0.8}
line_styles = {"solid": "Target", "dotted": "Target-UCB", "dashed": "Greedy"}

fig, axes = plt.subplots()

axes.plot(ucb_avg_01, c=list(colors)[0], ls=list(line_styles)[0])
axes.plot(target_ucb_avg_01, c=list(colors)[0], ls=list(line_styles)[1])
axes.plot(greedy_avg_01, c=list(colors)[0], ls=list(line_styles)[2])

axes.plot(ucb_avg_04, c=list(colors)[1], ls=list(line_styles)[0])
axes.plot(target_ucb_avg_04, c=list(colors)[1], ls=list(line_styles)[1])
axes.plot(greedy_avg_04, c=list(colors)[1], ls=list(line_styles)[2])

axes.plot(ucb_avg_08, c=list(colors)[2], ls=list(line_styles)[0])
axes.plot(target_ucb_avg_08, c=list(colors)[2], ls=list(line_styles)[1])
axes.plot(greedy_avg_08, c=list(colors)[2], ls=list(line_styles)[2])

patches = []
for key, value in colors.items():
  patches.append(mlines.Line2D([], [], c=key, ls="solid", label=f"Δ = {value}"))

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
