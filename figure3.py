import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx

import utils

np.random.seed(42)


nb_episode = 1000  # @param
nb_run = 100  # @param
nb_agent = 20
bandits = [utils.Bandit(nb_action=10) for _ in range(nb_run)]
to_plot = []

clique_neighborhood = np.ones((nb_agent,nb_agent), dtype=np.bool)
clique_neighborhood[range(nb_agent), range(nb_agent)] = False

chain_neighborhood = np.zeros((nb_agent, nb_agent), dtype=np.bool)
for i in range(nb_agent):
  if i != 0:
    chain_neighborhood[i, i-1] = True
  if i+1 != nb_agent:
    chain_neighborhood[i, i+1] = True

loop_neighborhood = np.zeros((nb_agent, nb_agent), dtype=np.bool)
for i in range(nb_agent):
  loop_neighborhood[i, (i - 1) % 20] = True
  loop_neighborhood[i, (i + 1) % 20] = True

random_neighborhood = np.zeros((nb_agent, nb_agent), dtype=np.bool) # according to Erdős–Rényi model
p = 0.5
for i in range(nb_agent):
  for j in range(i):
    if np.random.random() >= p :
      random_neighborhood[i][j] = True
      random_neighborhood[j][i] = True


smallWorld_neighborhood = nx.watts_strogatz_graph(n=nb_agent, k=4, p=0.5)
smallWorld_neighborhood = nx.to_numpy_array(smallWorld_neighborhood).astype(np.bool)

_, res_clique = utils.compute_result(clique_neighborhood, nb_run, nb_episode, bandits)
to_plot.append(res_clique)

_, res_chain = utils.compute_result(chain_neighborhood, nb_run, nb_episode, bandits)
to_plot.append(res_chain)

_, res_loop = utils.compute_result(loop_neighborhood, nb_run, nb_episode, bandits)
to_plot.append(res_loop)

_, res_random = utils.compute_result(random_neighborhood, nb_run, nb_episode, bandits)
to_plot.append(res_random)

_, res_smallWorld = utils.compute_result(smallWorld_neighborhood, nb_run, nb_episode, bandits)
to_plot.append(res_smallWorld)


ucb_regrets_cumul = []
for run in range(nb_run):
  bandit = bandits[run]
  if run % 100 == 0:
    print(run)

  ucb = utils.UCB(bandit)
  _, regrets = ucb.run(nb_episode)
  ucb_regrets_cumul.append(regrets)

ucb_full_avg = np.average(ucb_regrets_cumul, axis=0)
to_plot.append(ucb_full_avg)

# Plot
fig, axes = plt.subplots()

colors = {"tab:red": "Clique", "tab:orange": "Chain", "tab:green": "Loop", "tab:purple": "Random",
          "tab:brown": "Small-world", "tab:blue": "Single UCB"}

for i, plot in enumerate(to_plot):
  color = list(colors)[i]
  plt.plot(plot, c=color)

patches = []
for key, value in colors.items():
  patches.append(mlines.Line2D([], [], c=key, ls="solid", label=value))

legend = plt.legend(handles=patches, loc=2)

axes.add_artist(legend)

plt.ylabel("Cumulative regret")
plt.xlabel("Episodes")
plt.show()