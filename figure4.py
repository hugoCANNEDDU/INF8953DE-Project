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

bandit = utils.Bandit(mu_star, delta_mu, 1)

eq_dict = {"A": 0, "B": 1}
human_data_path = "./human_bandit_dataset/single_humans"
human_regrets = []
human_actions = []
for i in range(1,4):
  human_full = pd.read_csv(os.path.join(human_data_path, f"single_human_{i}_full_results.csv"), header=1)
  human_play = pd.read_csv(os.path.join(human_data_path, f"single_human_{i}_plays.csv"), header=1)
  human_regrets.append(utils.compute_regret(bandit, human_full["Result"]))
  human_actions.append(human_play.replace(eq_dict).to_numpy().squeeze())

target_ucb_regrets = []
for actions in human_actions :
  target_ucb_regrets_cumul = []
  for i in range(nb_run):
    bandit = utils.Bandit(mu_star, delta_mu, 1)
    target_ucb = utils.TargetUCB(bandit)
    if i % 100 == 0:
      print(i)
    target_ucb_regret = target_ucb.run(actions, nb_episode)
    target_ucb_regrets_cumul.append(target_ucb_regret)

  target_ucb_regrets.append(np.average(target_ucb_regrets_cumul, axis=0))

fig, axes = plt.subplots()

colors = {"tab:orange": "Human 1", "tab:green": "Human 2", "tab:red": "Human 3"} #, "tab:blue": "Human 4"
line_styles = {"dashed": "Single Human", "solid": "Target-UCB"}

for i, target_ucb_regret in enumerate(target_ucb_regrets):
  color = list(colors)[i]
  ls = list(line_styles)[1]
  plt.plot(target_ucb_regret, ls=ls, c=color)

for i, human_regret in enumerate(human_regrets):
  color = list(colors)[i]
  ls = list(line_styles)[0]
  plt.plot(human_regret, ls=ls, c=color)

patches = []
for key, value in colors.items():
  patches.append(mlines.Line2D([], [], c=key, ls="solid", label=value))

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