import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import utils

np.random.seed(42)

mu_star = 0.6  # @param
delta_mu = 0.2  # @param
nb_episode = 100  # @param
nb_run = 200  # @param
nb_agent = 4

# Target UCB clique
bandit = utils.Bandit(mu_star, delta_mu, 0)

clique_neighborhood = np.ones((nb_agent,nb_agent), dtype=np.bool)
clique_neighborhood[range(nb_agent), range(nb_agent)] = False
bandits = [utils.Bandit(mu_star, delta_mu, 0) for _ in range(nb_run)]
target_ucb_avg, target_ucb_full_avg = utils.compute_result(clique_neighborhood, nb_run, nb_episode, bandits)

# Human clique
human_clique_data_path = "./human_bandit_dataset/cliques"
human_ucb_regrets = [[] for _ in range(1)] # Change 1 to 2 for all data
for i in range(1, 2): # Change 2 to 3 for all data
  human_clique_full = pd.read_csv(os.path.join(human_clique_data_path, f"human_clique_{i}_full_results.csv"), header=1)
  results = ["", ".1", ".2", ".3"]
  for result in results:
    human_ucb_regrets[i-1].append(utils.compute_regret(bandit, human_clique_full[f"Result{result}"]))

human_ucb_regrets_avg = [np.average(data, axis=0) for data in human_ucb_regrets]

# Plot
fig, axes = plt.subplots()

colors = {"tab:red": "Target-UCB clique", "tab:green": "Human clique"} #, "tab:blue": "Human clique 2"
line_styles = {"dashed": "Individual Agents", "solid": "Clique average"}

# Human clique
for i, human_ucb_regrets in enumerate(human_ucb_regrets):
  for individual_human_ucb_regrets in human_ucb_regrets:
    color = list(colors)[1 + i]
    ls = list(line_styles)[0]
    plt.plot(individual_human_ucb_regrets, ls=ls, c=color)

for i, human_ucb_regret_avg in enumerate(human_ucb_regrets_avg):
  plt.plot(human_ucb_regret_avg, ls=list(line_styles)[1], c=list(colors)[1+i])

# Target UCB clique
for individual_target_ucb_avg in target_ucb_avg:
  color = list(colors)[0]
  ls = list(line_styles)[0]
  plt.plot(individual_target_ucb_avg, ls=ls, c=color)

plt.plot(target_ucb_full_avg, ls=list(line_styles)[1], c=list(colors)[0])

patches = []
for key, value in colors.items():
  patches.append(mlines.Line2D([], [], c=key, ls="solid", label=value))

lines = []
for key, value in line_styles.items():
  lines.append(mlines.Line2D([], [], c="black", ls=key, label=value))

legend1 = plt.legend(handles=patches, loc=2)
legend2 = plt.legend(handles=lines, loc=2, bbox_to_anchor=(0, 0.85))

axes.add_artist(legend1)
axes.add_artist(legend2)
plt.ylabel("Cumulative regret")
plt.xlabel("Episodes")
plt.show()