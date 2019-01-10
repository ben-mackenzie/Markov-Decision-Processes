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

d_rands = [0.5 + i*0.025 for i in range(20)]
epsilon = 0.1
gamma = 0.9
threshold = 9 # epsilon * (1 - gamma) / gamma

ql_times = []
ql_iters_list = []

for d_rand in d_rands:
       grid_mdp = GridMDP(grid, terminals, d_rand = d_rand, )

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
       print("movement probability: ", d_rand)
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
plt.plot(d_rands, ql_times)
plt.xlabel('Probability of Intended Movement')
plt.ylabel('Seconds')
plt.title('Q Learning: Stochasticity vs Time to Convergence')
plt.show()

plt.figure(2)
plt.plot(d_rands, ql_iters_list)
plt.xlabel('Probability of Intended Movement')
plt.ylabel('Iterations')
plt.title('Q Learning: Stochasticity vs Iterations to Convergence')
plt.show()