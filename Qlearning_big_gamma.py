from rl import *
from mdp import *
import matplotlib.pyplot as plt

alpha = lambda n: 60./(59+n)
Rplus = 10
Ne = 10000

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

gammas = [.95]
threshold = 100

ql_times = []
ql_iters_list = []

for gamma in gammas:
       grid_mdp = GridMDP(grid, terminals, gamma = gamma)

       # Value Iteration
       U_VI, vi_time, vi_iters = value_iteration(grid_mdp)
       pi_VI = best_policy(grid_mdp, U_VI)

       # Q Learning
       q_agent = QLearningAgent(grid_mdp, Ne, Rplus, alpha)
       i_VI = 1
       t1 = time.time()
       pi_QL = None

       # train agent
       converged = False
       while converged == False:
           # run single trials
           run_single_trial(q_agent, grid_mdp)
           U = defaultdict(lambda: -100.)  # Very Large Negative Value for Comparison see below.
           for state_action, value in q_agent.Q.items():
               state, action = state_action
               if U[state] < value:
                   U[state] = value

           # check for convergence
           delta = 0
           for s in grid_mdp.states:
               delta = max(delta, abs(U_VI[s] - U[s]))
               #print(delta, threshold)
           if delta <= threshold:
               t2 = time.time()
               convergence_time = t2 - t1
               converged = True
           else:
               i_VI += 1
       print("iters: ", i_VI)
       print("gamma: ", gamma)
       pi_QL = best_policy(grid_mdp, U)

       # compare policies
       for row in grid_mdp.to_arrows(pi_VI):
           print(row)
       print(" ")
       for row in grid_mdp.to_arrows(pi_QL):
           print(row)

       # append time and iter data
       t2 = time.time()
       ql_times.append(t2 - t1)
       ql_iters_list.append(i_VI)

plt.figure(1)
plt.plot(gammas, ql_times)
plt.xlabel('Gamma')
plt.ylabel('Seconds')
plt.title('Q Learning: Gamma vs Time to Convergence')
plt.show()

plt.figure(2)
plt.plot(gammas, ql_iters_list)
plt.xlabel('Gamma')
plt.ylabel('Iterations')
plt.title('Q Learning: Gamma vs Iterations to Convergence')
plt.show()