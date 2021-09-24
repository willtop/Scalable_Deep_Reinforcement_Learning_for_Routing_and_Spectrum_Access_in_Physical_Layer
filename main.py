# This script contains our novel deep reinforcement learning network model implementation
# for the work "Scalable Deep Reinforcement Learning for Routing and Spectrum Access in Physical Layer", 
# available at arxiv.org/abs/2012.11783.

# For any reproduce, further research or development, please kindly cite our paper (TCOM Journal version upcoming soon):
# @misc{rl_routing,
#    author = "W. Cui and W. Yu",
#    title = "Scalable Deep Reinforcement Learning for Routing and Spectrum Access in Physical Layer",
#    month = dec,
#    year = 2020,
#    note = {[Online] Available: https://arxiv.org/abs/2012.11783}
# }

# Main script to train the agent

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange
import sys
import evaluate
sys.path.append("DQN/")
import adhoc_wireless_net
import agent
from replay_memory import Prioritized_Replay_Memory, Uniform_Replay_Memory
from system_parameters import *

INITIAL_EXPLORE_STEPS = int(30e3)
EPSILON_GREEDY_STEPS = int(300e3)
FINAL_CONVERGE_STEPS = int(50e3)
REPLAY_MEMORY_SIZE = int(100e3)
EVALUATE_FREQUENCY = int(5e3)
TARGET_NET_SYNC_FREQUENCY = int(5e3)

# Linear scheduler for hyper-parameters during training. Modified from the code from openai baselines:
#     https://github.com/openai/baselines/tree/master/baselines
class LinearSchedule():
    def __init__(self, initial_val, final_val, schedule_timesteps):
        self.initial_val = initial_val
        self.final_val = final_val
        self.schedule_timesteps = schedule_timesteps
    def value(self, timestep):
        fraction = max(min(float(timestep) / self.schedule_timesteps, 1.0),0.0)
        return self.initial_val + fraction * (self.final_val - self.initial_val)

if (__name__ == "__main__"):
    print("Training DQN with {} replay memory".format(REPLAY_MEMORY_TYPE))
    adhocnet = adhoc_wireless_net.AdHoc_Wireless_Net()
    agents = [agent.Agent(adhocnet, i) for i in range(adhocnet.n_flows)]
    memory = Prioritized_Replay_Memory(REPLAY_MEMORY_SIZE) if REPLAY_MEMORY_TYPE=="Prioritized" \
                else Uniform_Replay_Memory(REPLAY_MEMORY_SIZE)

    metrics = [] # store two metrics [Q_Loss, routing rate performance]
    best_min_bottleneck_rate = -np.inf
    policy_epsilon = LinearSchedule(initial_val=1.0, final_val=0.1, schedule_timesteps=EPSILON_GREEDY_STEPS)
    priority_ImpSamp_beta = LinearSchedule(initial_val=0.4, final_val=1.0, schedule_timesteps=EPSILON_GREEDY_STEPS)
    for i in trange(1, INITIAL_EXPLORE_STEPS+EPSILON_GREEDY_STEPS+FINAL_CONVERGE_STEPS+1):
        # refresh the layout
        adhocnet.update_layout()
        for agent in agents[:-1]:
            while not agent.flow.destination_reached():
                agent.route_close_neighbor_closest_to_destination()
        while not agents[-1].flow.destination_reached():
            # final settlement for Monte-Carlo estimation based learning
            epsilon_val = 0 if i>(INITIAL_EXPLORE_STEPS+EPSILON_GREEDY_STEPS) \
                            else policy_epsilon.value(i-1-INITIAL_EXPLORE_STEPS)
            agents[-1].route_epsilon_greedy(epsilon=epsilon_val)
        agents[-1].process_links(memory)
        for agent in agents:
            agent.reset()
        if i >= INITIAL_EXPLORE_STEPS:  # Have gathered enough experiences, start training the agents
            Q_loss = agents[-1].train(memory, priority_ImpSamp_beta.value(i-1-INITIAL_EXPLORE_STEPS))
            assert not np.isnan(Q_loss)
            if (i % TARGET_NET_SYNC_FREQUENCY == 0):
                agents[-1].sync_target_network()
            if (i % EVALUATE_FREQUENCY == 0):
                for agent in agents[:-1]: # load the currently trained model parameters to evaluate
                    agent.sync_main_network_from_another_agent(agents[-1])
                eval_results = evaluate.evaluate_routing(adhocnet, agents, "DDQN_Q_Novel", n_layouts=50)
                min_bottleneck_rate_avg = np.mean(np.min(eval_results[:,:,0],axis=1))/1e6
                print("Q loss: {:.3f}; Min Bottleneck Rate(mbps): {:.3g}".format(Q_loss, min_bottleneck_rate_avg))
                metrics.append([i, Q_loss, min_bottleneck_rate_avg])
                if best_min_bottleneck_rate < min_bottleneck_rate_avg:
                    agents[-1].save_dqn_model()
                    best_min_bottleneck_rate = min_bottleneck_rate_avg
    agents[-1].visualize_non_zero_rewards(memory)
    fig, axes = plt.subplots(1,2)
    fig.suptitle("Training Progress for Individual Agent In Sequential Routing")
    metrics = np.array(metrics)
    x_vals = metrics[:, 0] / 1e3
    axes[0].set_xlabel("Number of Layouts (1e3)")
    axes[0].set_ylabel("Q Loss")
    axes[0].plot(x_vals, np.log(metrics[:,1]))
    axes[1].set_xlabel("Number of Layouts Trained (1e3)")
    axes[1].set_ylabel("Min Bottleneck Rate (mbps)")
    axes[1].plot(x_vals, metrics[:, 2])
    plt.show()

    print("Script Finished Successfully!")
