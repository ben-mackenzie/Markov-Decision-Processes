# Library source: https://github.com/aimacode/aima-python
### ROBTO DOG POOP
### Goal: get home without rolling over dog poop

from mdp import *
import copy
import matplotlib.pyplot as plt

### TODO ###
# calculate mean reward and plot vs iters/episodes?
# scale up grid enough to show differences between VI/PI/QL performance in small vs large?

### QUESTIONS ###
# Did the algo find a solution?
# Is it the optimal solution?
# Are there multiple optimal solutions?
# Q-learning: how did I "map the exploration"?  How did I "tweak my hyperparameters"?

# build grid, assigning reward or penalty to each cell and setting blocked squares to 'None'
small_grid = [[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
        [-1, -0.04, -0.04, -0.04, -0.04, -1, -0.04, +10, -0.04],
       [-0.04, -0.04, -0.04, -1, -0.04, -0.04, -0.04, -1, -0.04],
       [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04]]


factor = 3
grid = [[-0.04 for c in range(len(small_grid[0])*factor)] for r in range(len(small_grid)*factor)]
terminals = []
for r in range(len(small_grid)):
       for c in range(len(small_grid[0])):
              cell = small_grid[r][c]
              if cell != -0.04:
                     grid[r*factor][c*factor] = small_grid[r][c]
                     if cell == 1 or cell == -1:
                            row = r*factor - len(grid)
                            col = c*factor

terminals = [(0, 8), (9, 5), (15, 8), (21, 5), (21, 8)]


gammas = [0.5 + i*0.025 for i in range(20)]
vi_times = []
vi_iters_list = []
pi_times = []
pi_iters_list = []
for gamma in gammas:
       grid_mdp = GridMDP(grid, terminals, gamma = gamma)

       # Value Iteration
       vi_U, vi_time, vi_iters = value_iteration(grid_mdp)
       vi_pi = best_policy(grid_mdp, vi_U)
       vi_times.append(vi_time)
       vi_iters_list.append(vi_iters)
       vi_visualized = grid_mdp.to_arrows(vi_pi)
       print("Value Iteration Policies, gamma =", gamma)
       for row in vi_visualized:
              print(row)
       print(" ")

       # Policy Iteration
       pi_pi, pi_time, pi_iters = policy_iteration(grid_mdp)
       pi_times.append(pi_time)
       pi_iters_list.append(pi_iters)
       pi_visualized = grid_mdp.to_arrows(pi_pi)
       print("Policy Iteration Policies, gamma =", gamma)
       for row in pi_visualized:
              print(row)
       print(" ")


print(vi_times)
print(vi_iters_list)

print(pi_times)
print(pi_iters_list)


plt.figure(1)
plt.plot(gammas, pi_times, label = "Policy Iteration")
plt.plot(gammas, vi_times, label = "Value Iteration")
plt.legend()
plt.xlabel('Gamma')
plt.ylabel('Seconds')
plt.title('VI/PI: Gamma vs Time to Convergence')
plt.show()

plt.figure(2)
plt.plot(gammas, pi_iters_list, label = "Policy Iteration")
plt.plot(gammas, vi_iters_list, label = "Value iteration")
plt.legend()
plt.xlabel('Gamma')
plt.ylabel('Iterations')
plt.title('VI/PI: Gamma vs Iterations to Convergence')
plt.show()




