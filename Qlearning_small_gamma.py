from rl import *
from mdp import *
import matplotlib.pyplot as plt

alpha = lambda n: 60./(59+n)
Rplus = 10
Ne = 500

grid = [[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
        [-1, -0.04, -0.04, -0.04, -0.04, -1, -0.04, +10, -0.04],
       [-0.04, -0.04, -0.04, -1, -0.04, -0.04, -0.04, -1, -0.04],
       [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04]]

terminals = [(0, 2), (3, 1), (5, 2), (7, 1), (7, 2)]

gammas = [0.5 + i*0.025 for i in range(20)]
threshold = 9

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
           U = defaultdict(lambda: -500.)  # Very Large Negative Value for Comparison see below.
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