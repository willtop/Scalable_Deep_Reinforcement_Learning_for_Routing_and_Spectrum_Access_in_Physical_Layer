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


# Benchmarks for routing
# Each method accepts one data flow object
import numpy as np
from system_parameters import *

def route_close_neighbor_closest_to_destination(agent):
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    assert np.size(available_bands) > 0
    agent_dist_to_dest = agent.adhocnet.nodes_distances[agent.flow.frontier_node, agent.flow.dest]
    while True:
        closest_neighbors = agent.get_closest_neighbors(available_bands) # every neighor returned has at least one band available
        states = agent.get_state(available_bands, closest_neighbors)
        dists_to_dest = agent.adhocnet.nodes_distances[closest_neighbors, agent.flow.dest]
        dists_to_dest = np.append(dists_to_dest, agent_dist_to_dest)
        action_index = np.argmin(dists_to_dest) # get the closest to terminal node one by one
        if action_index < agent.n_actions - 1: # found one neighbor closer to the destination
            next_hop = closest_neighbors[action_index]
            interferences = agent.obtain_interference_from_states(states, action_index)
            band_index = np.argmin(interferences + agent.adhocnet.nodes_on_bands[available_bands,next_hop]*1e6) # use the power exposed feature to determine the optimal band
            agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[band_index],
                                    rx=next_hop, state=states[band_index], action=action_index)
            break
        # reprobe as all the closest neighbors are further away from the destination
        agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[0],
                                rx=agent.flow.frontier_node, state=states[0], action=agent.n_actions-1)
        # This benchmark is not to be mixed with DQN routing, thus can exclude these nodes for further consideration
        agent.flow.exclude_nodes = np.append(agent.flow.exclude_nodes, closest_neighbors)
    return

def route_random(agent):
    # Generate a random node as next hop, transmit to it using a random band
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    assert np.size(available_bands) > 0
    band = np.random.choice(available_bands)
    band_index = np.where(band == available_bands)[0][0]
    # Generate a random node that's also available on the band
    available_nodes = np.where(agent.adhocnet.nodes_on_bands[band]==0)[0]
    available_nodes = agent.remove_nodes_excluded(available_nodes, agent.flow.exclude_nodes)
    assert np.size(available_nodes) > 0, "At least the destination node has to be there!"
    next_hop = np.random.choice(available_nodes)
    # Since random routing is to be used together with DQN, can't abort any nodes here
    exclude_nodes_original = np.copy(agent.flow.exclude_nodes)
    while True: # reproduce it as a sequence of actions by the agent
        # only probe on randomly selected bands (would be faster to find the node than probe over all available bands to the agent)
        closest_neighbors = agent.get_closest_neighbors(band)
        states = agent.get_state(available_bands, closest_neighbors)
        if next_hop in closest_neighbors:
            action_index = np.where(closest_neighbors == next_hop)[0][0]
            agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=band,
                                    rx=next_hop, state=states[band_index], action=action_index)
            break
        # reprobe (only store reprobe on the randomly selected band)
        agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=band,
                                rx=agent.flow.frontier_node, state=states[band_index], action=agent.n_actions-1)
        agent.flow.exclude_nodes = np.append(agent.flow.exclude_nodes, closest_neighbors)
    # restore the exclude nodes list
    agent.flow.exclude_nodes = np.append(exclude_nodes_original, next_hop)
    return

def route_close_neighbor_under_lowest_power(agent):
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    closest_neighbors = agent.get_closest_neighbors(available_bands)
    states = agent.get_state(available_bands, closest_neighbors)
    interferences = [agent.obtain_interference_from_states(states, i)+agent.adhocnet.nodes_on_bands[available_bands,close_neighbor]*1e6
                            for i, close_neighbor in enumerate(closest_neighbors)]
    interferences = np.transpose(interferences)
    assert np.shape(interferences) == (np.size(available_bands), np.size(closest_neighbors))
    band_index, action_index = np.unravel_index(np.argmin(interferences), np.shape(interferences))
    agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[band_index],
                            rx=closest_neighbors[action_index], state=states[band_index], action=action_index)
    return

# Route towards the neighbor with the strongest wireless channel from the frontier node
# Since using path-loss as the channel model, this is equivalent to selecting the closest neighbor.
# If the closest neighbor has more than one band available, go to the band with the lowest interference
def route_strongest_neighbor(agent):
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    closest_neighbors = agent.get_closest_neighbors(available_bands)
    states = agent.get_state(available_bands, closest_neighbors)
    action_index = 0 # choose the closest neighbor
    interferences = agent.obtain_interference_from_states(states, action_index)
    band_index = np.argmin(interferences+agent.adhocnet.nodes_on_bands[available_bands,closest_neighbors[action_index]]*1e6)
    agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[band_index],
                            rx=closest_neighbors[action_index], state=states[band_index], action=action_index)
    return

# Search within closest neighbors every available neighbor-band combination
def route_close_neighbor_with_largest_forward_rate(agent):
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    closest_neighbors = agent.get_closest_neighbors(available_bands)
    states = agent.get_state(available_bands, closest_neighbors)
    rates = []
    for neighbor in closest_neighbors:
        rates_one_neighbor = []
        for band in available_bands:
            if agent.adhocnet.nodes_on_bands[band, neighbor] == 1:
                rates_one_neighbor.append(-1)
            else:
                agent.adhocnet.powers[band, agent.flow.frontier_node] = TX_POWER # temporaily set
                rate, _, _ = agent.adhocnet.compute_link(tx=agent.flow.frontier_node, band=band, rx=neighbor)
                rates_one_neighbor.append(rate)
                agent.adhocnet.powers[band, agent.flow.frontier_node] = 0
        rates.append(rates_one_neighbor)
    rates = np.transpose(rates); assert np.shape(rates) == (np.size(available_bands), np.size(closest_neighbors))
    assert np.max(rates) > 0, "Has to be one node/band available"
    band_index, action_index = np.unravel_index(np.argmax(rates), np.shape(rates))
    agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[band_index],
                            rx=closest_neighbors[action_index], state=states[band_index], action=action_index)
    return

# If all neighbors are behind the agent (i.e. the angle difference is greater than 90 degrees) then reprobe
def route_close_neighbor_best_forwarding_direction(agent):
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    angle_self_to_dest = agent.adhocnet.obtain_angle(node_1=agent.flow.frontier_node, node_2=agent.flow.dest)
    while True:
        closest_neighbors = agent.get_closest_neighbors(available_bands)
        states = agent.get_state(available_bands, closest_neighbors)
        angles = []
        for neighbor in closest_neighbors:
            angle_self_to_neighbor = agent.adhocnet.obtain_angle(node_1=agent.flow.frontier_node, node_2=neighbor)
            angles.append(agent.compute_angle_offset(angle_1=angle_self_to_dest, angle_2=angle_self_to_neighbor))
        angles.append(np.pi/2) # make sure we don't look for any neighbor going backwards
        action_index = np.argsort(angles)[0]
        if action_index < agent.n_actions - 1:  # a neighbor with forward angle
            interferences = agent.obtain_interference_from_states(states, action_index)
            band_index = np.argmin(interferences + agent.adhocnet.nodes_on_bands[available_bands, closest_neighbors[action_index]]*1e6) # use the power exposed feature to determine the optimal band
            agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[band_index],
                                    rx=closest_neighbors[action_index], state=states[band_index], action=action_index)
            break
        # reprobe as all the closest neighbors are with backward angle
        agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[0],
                                    rx=agent.flow.frontier_node, state=states[0], action=agent.n_actions-1)
        agent.flow.exclude_nodes = np.append(agent.flow.exclude_nodes, closest_neighbors)
    return

def route_destination_directly(agent):
    assert agent.flow.frontier_node == agent.flow.src
    available_bands = agent.adhocnet.get_available_bands(agent.flow.src)
    # For here, just directly compute power exposed field
    interferences = []
    for band in available_bands:
        _, _, interference = agent.adhocnet.compute_link(tx=agent.flow.src, band=band, rx=agent.flow.dest)
        interferences.append(interference)
    band_index = np.argmin(interferences) # For the destination, no need to check for band availability
    # Just append pseudo state and action (not used in agent training)
    agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.src, band=available_bands[band_index],
                            rx=agent.flow.dest, state=np.zeros(agent.state_dim), action=0)
    return
