import time
import numpy as np
import os
from argparse import Namespace
from pprint import pprint
from random import *
from datetime import datetime
from collections import deque
import psutil

import torch

from torch.utils.tensorboard import SummaryWriter
# In Flatland you can use custom observation builders and predicitors
# Observation builders generate the observation needed by the controller
# Preditctors can be used to do short time prediction which can help in avoiding conflicts in the network
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv, GlobalObsModifiedRailEnv, TreeTimetableObservation
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
# Import the railway generators   
from flatland.envs.custom_rail_generator import rail_custom_generator
from flatland.envs.rail_env_utils import delay_a_train, make_a_deterministic_interruption
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
# Import the schedule generators
from flatland.envs.custom_schedule_generator import custom_schedule_generator
from flatland.envs.plan_to_follow_utils import action_to_do, divide_trains_in_station_rails, control_timetable
# Import the different structures needed
from configuration import railway_example, stations, timetable_example, example_training
# Import the agent class
from flatland.envs.agent import RandomAgent
from flatland.envs.step_utils.states import TrainState
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.utils.deadlock_check import find_and_punish_deadlock

from flatland.utils.agent_action_config import get_flatland_full_action_size, get_action_size, map_actions, \
    map_action, \
    set_action_size_reduced, set_action_size_full, convert_default_rail_env_action

from flatland.utils.timer import Timer
from flatland.utils.observation_utils import normalize_global_observation, normalize_observation
from reinforcement_learning.dddqn_policy import DDDQNPolicy
# Import training and observation parameters
from parameters import training_params, obs_params


import matplotlib.pyplot as plt
import matplotlib.animation
plt.rcParams["animation.html"] = "jshtml"

def display_episode(frames):
    fig, ax = plt.subplots(figsize=(12,12))
    imgplot = plt.imshow(frames[0])
    def animate(i):
        imgplot.set_data(frames[i])
    animation = matplotlib.animation.FuncAnimation(fig, animate, frames=len(frames))
    return animation

def check_conflicts(env):
    for a in range(len(env.agents)):
        if env.agents[a].state_machine.st_signals.movement_conflict == True:
            return True

def choose_a_random_training_configuration(env, max_steps):
    if example_training == 'training0':
        case = 0
        if case == 0:
            env.agents[1].initial_position = (6,8)
            env.agents[2].initial_position = (5,8)
            make_a_deterministic_interruption(env.agents[1], max_steps)
            make_a_deterministic_interruption(env.agents[2], max_steps)
            return    
        elif case == 1:
            env.agents[1].initial_position = (6,15)
            env.agents[2].initial_position = (5,15)
            make_a_deterministic_interruption(env.agents[1], max_steps)
            make_a_deterministic_interruption(env.agents[2], max_steps)
            return
        elif case == 2:
            env.agents[1].initial_position = (6,8)
            make_a_deterministic_interruption(env.agents[1], max_steps)
            env.agents[2].malfunction_handler.malfunction_down_counter = max_steps
            return       
        elif case == 3:
            env.agents[1].malfunction_handler.malfunction_down_counter = max_steps
            env.agents[2].initial_position = (5,15)
            make_a_deterministic_interruption(env.agents[2], max_steps)
            return
    else:
        env.agents[1].initial_position = (5,8)
        make_a_deterministic_interruption(env.agents[1], max_steps)
        

def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼", "↓"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer

def eval_policy(env, tree_observation, policy, train_params, obs_params):
    n_eval_episodes = train_params.n_evaluation_episodes
    max_steps = env._max_episode_steps

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(n_eval_episodes):
        score = 0.0

        agent_obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        policy.reset(env)
        final_step = 0

        if train_params.eval_render:
            # Setup renderer
            env_renderer = RenderTool(env, gl="PGL",
                                      show_debug=True,
                                      agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS)
            env_renderer.set_new_rail()

        policy.start_episode(train=False)
        for step in range(max_steps - 1):
            policy.start_step(train=False)
            for agent in env.get_agent_handles():
                action = convert_default_rail_env_action(RailEnvActions.DO_NOTHING)
                if info['action_required'][agent]:
                    action = policy.act(agent, agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})
            policy.end_step(train=False)
            agent_obs, all_rewards, dones, info = env.step(map_actions(action_dict))

            for agent in env.get_agent_handles():
                dones[agent] = env.agents[agent].state == TrainState.DONE

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if dones['__all__']:
                break

            # Render an episode at some interval
            if train_params.eval_render:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=True,
                    show_predictions=False
                )

        policy.end_episode(train=False)
        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(dones[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

        if train_params.eval_render:
            env_renderer.close_window()
            print(
                " ✅ {} : score {:.3f} done {:.1f}%".format(episode_idx, np.mean(normalized_score), completion * 100.0))

    print(" ✅ Eval: score {:.3f} done {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0))

    return scores, completions, nb_steps

#########################################################
# Parameters that should change the results:

# eps_decay 
# step_maximum_penality
# % used in the sparse reward
# penalty given in the function find_and_punish_deadlock
#########################################################

render = True

# The specs for the custom railway generation are taken from structures.py file
specs = railway_example

widht = len(specs[0])
height = len(specs)

stations_position = []
    
# Positions of the stations
for i in range(len(stations)):
    stations_position.append(stations[i].position)

# Timetable conteins the station where the train should pass, from starting station to aim, and conteins the time at which
# each train has to pass in the station, the last number represent the velocity of train (high velocity, intercity or regional)
# Each row represent a different train

print('------ Calculating the timetable')
print()
timetable = timetable_example

# Number of agents is the rows of the timetable
num_of_agents = len(timetable)

# Check if the timetable is feaseble or not, the function is in schedule_generators
# A timetable is feaseble if the difference of times between two stations is positive and let the trains to reach the successive station
# if two stations are very distant from each other the difference of times can't be very small
seed = 2

# Generating the railway topology, with stations
# Arguments of the generator (specs of the railway, position of stations, timetable)
rail_custom = rail_custom_generator(specs, stations_position, timetable)

transition_map_example, agent_hints = rail_custom(widht, height, num_of_agents)

divide_trains_in_station_rails(timetable, transition_map_example)

control_timetable(timetable,transition_map_example)

print('Station | Departure time |  Train id')
print('-------------------------------------')
for i in range(len(timetable)):
    for j in range(len(timetable[i][0])):
        print(timetable[i][0][j].name, ' | ' ,timetable[i][1][j], '  |  ', timetable[i][2].id)
        print('-------------------------------------')
        
 
time.sleep(3)

# We can now initiate the schedule generator with the given speed profiles
schedule_generator_custom = custom_schedule_generator(timetable = timetable)

###### TRAINING PARAMETERS #######
eps_start = training_params.eps_start
eps_end = training_params.eps_end
eps_decay = training_params.eps_decay
n_episodes = training_params.n_episodes
checkpoint_interval = training_params.checkpoint_interval
n_eval_episodes = training_params.n_evaluation_episodes
restore_replay_buffer = training_params.restore_replay_buffer
save_replay_buffer = training_params.save_replay_buffer
skip_unfinished_agent = training_params.skip_unfinished_agent

max_steps = 250
num_of_conflict = 0
 # Unique ID for this training
now = datetime.now()
training_id = now.strftime('%y%m%d%H%M%S')

####### OBSERVATION BUILDER ########
observation_parameters = Namespace(**obs_params)

observation_tree_depth = observation_parameters.observation_tree_depth
observation_radius = observation_parameters.observation_radius
observation_max_path_depth = observation_parameters.observation_max_path_depth

# Observation builder
predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
Observer = TreeTimetableObservation(max_depth=observation_tree_depth, predictor=predictor)


stochastic_data = MalfunctionParameters(
    malfunction_rate = 0,  # Rate of malfunction occurence
    min_duration = 15,  # Minimal duration of malfunction
    max_duration = 40  # Max duration of malfunction
)

malfunction_generator = ParamMalfunctionGen(stochastic_data)

env = RailEnv(  width= widht,
                height= height,
                rail_generator = rail_custom,
                line_generator=schedule_generator_custom,
                number_of_agents= num_of_agents,
                malfunction_generator = malfunction_generator,
                obs_builder_object=Observer,
                remove_agents_at_target=True,
                record_steps=True,
                max_episode_steps = max_steps - 1
                )
env.reset()
env_renderer = RenderTool(env,
                          screen_height=1080,
                          screen_width=1080)  # Adjust these parameters to fit your resolution


# This thing is importand for the RL part, initialize the agent with (state, action) dimension
# Initialize the agent with the parameters corresponding to the environment and observation_builder

n_features_per_node = env.obs_builder.observation_dim
n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
state_size = n_features_per_node * n_nodes

n_agents = env.get_num_agents()
action_size = get_action_size()

action_count = [0] * action_size
action_dict = dict()
agent_obs = [None] * n_agents
agent_prev_obs = [None] * n_agents
agent_prev_action = [2] * n_agents
update_values = [False] * n_agents

# Smoothed values used as target for hyperparameter tuning
smoothed_eval_normalized_score = -1.0
smoothed_eval_completion = 0.0

scores_window = deque(maxlen=checkpoint_interval)  # todo smooth when rendering instead
completion_window = deque(maxlen=checkpoint_interval)
deadlocked_window = deque(maxlen=checkpoint_interval)

set_action_size_reduced()

train_params = training_params

policy = DDDQNPolicy(state_size, action_size, train_params,
                     enable_delayed_transition_push_at_episode_end=False,
                     skip_unfinished_agent=skip_unfinished_agent)

# Load existing policy
if train_params.load_policy != "":
    policy.load(train_params.load_policy)

# Loads existing replay buffer
if restore_replay_buffer:
    try:
        policy.load_replay_buffer(restore_replay_buffer)
        policy.test()
    except RuntimeError as e:
        print(
            "\n🛑 Could't load replay buffer, were the experiences generated using the same tree depth?")
        print(e)
        exit(1)

print("\n💾 Replay buffer status: {}/{} experiences".format(len(policy.memory.memory),
                                                            train_params.buffer_size))

hdd = psutil.disk_usage('/')
if save_replay_buffer and (hdd.free / (2 ** 30)) < 500.0:
    print(
        "⚠️  Careful! Saving replay buffers will quickly consume a lot of disk space. You have {:.2f}gb left."
            .format(hdd.free / (2 ** 30)))




# TensorBoard writer
writer = SummaryWriter(comment="_" +
                                train_params.policy + "_" +
                                train_params.use_observation + "_" +
                                train_params.action_size)

training_timer = Timer()
training_timer.start()

print(
    "\n🚉 Training {} trains on {}x{} grid for {} episodes, evaluating {} trains on {} episodes every {} episodes. "
    "Training id '{}'.\n".format(
        env.get_num_agents(),
        widht, height,
        n_episodes,
        env.get_num_agents(),
        n_eval_episodes,
        checkpoint_interval,
        training_id
    ))

for episode_idx in range(n_episodes + 1):
    reset_timer = Timer()
    policy_start_episode_timer = Timer()
    policy_start_step_timer = Timer()
    policy_act_timer = Timer()
    env_step_timer = Timer()
    policy_shape_reward_timer = Timer()
    policy_step_timer = Timer()
    policy_end_step_timer = Timer()
    policy_end_episode_timer = Timer()
    total_episode_timer = Timer()
    # Reset environment
    total_episode_timer.start()

    action_count = [0] * get_flatland_full_action_size()
    agent_prev_obs = [None] * n_agents
    agent_prev_action = [convert_default_rail_env_action(RailEnvActions.STOP_MOVING)] * n_agents
    update_values = [False] * n_agents

    # Reset environment
    reset_timer.start()
    number_of_agents = n_agents
    agent_obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    policy.reset(env)
    reset_timer.end()

    if train_params.render:
        env_renderer = RenderTool(env, gl="PGL",
                                      show_debug=True,
                                      agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS)

        env_renderer.set_new_rail()

    score = 0
    nb_steps = 0
    metric = 0
    actions_taken = []
    
    # Build initial agent-specific observations
    for agent in env.get_agent_handles():
        agent_obs[agent] = normalize_observation(agent_obs[agent], observation_tree_depth, observation_radius=observation_radius)
    
    policy_start_episode_timer.start()
    policy.start_episode(train=True)
    policy_start_episode_timer.end()
    
    for step in range(max_steps - 1):
        
        # policy.start_step ---------------------------------------------------------------------------------------
        policy_start_step_timer.start()
        policy.start_step(train=True)
        policy_start_step_timer.end()
        
        # policy.act ----------------------------------------------------------------------------------------------
        policy_act_timer.start()
        action_dict = {}
        
        for agent_handle in policy.get_agent_handles(env):
            if info['action_required'][agent_handle]:
                update_values[agent_handle] = True
                action = policy.act(agent_handle, agent_obs[agent_handle], eps=eps_start)
                action_count[map_action(action)] += 1
                actions_taken.append(map_action(action))
                
            else:
                # An action is not required if the train hasn't joined the railway network,
                # if it already reached its target, or if is currently malfunctioning.
                update_values[agent_handle] = False
                action = convert_default_rail_env_action(RailEnvActions.DO_NOTHING)

            action_dict.update({agent_handle: action})
        policy_act_timer.end()
        
        # policy.end_step -----------------------------------------------------------------------------------------
        policy_end_step_timer.start()
        policy.end_step(train=True)
        policy_end_step_timer.end()
        
        # Environment step ----------------------------------------------------------------------------------------
        env_step_timer.start()
        next_obs, all_rewards, dones, info = env.step(map_actions(action_dict))
        for agent_handle in env.get_agent_handles():
            dones[agent_handle] = (env.agents[agent_handle].state == TrainState.DONE)
        env_step_timer.end()
        
        # policy.shape_reward -------------------------------------------------------------------------------------
        policy_shape_reward_timer.start()
        # Deadlock
        deadlocked_agents, all_rewards, = find_and_punish_deadlock(env, all_rewards,
                                                                       0)
        
        # The might requires a policy based transformation
        for agent_handle in env.get_agent_handles():
            all_rewards[agent_handle] = policy.shape_reward(agent_handle,
                                                            action_dict[agent_handle],
                                                            agent_obs[agent_handle],
                                                            all_rewards[agent_handle],
                                                            dones[agent_handle],
                                                            deadlocked_agents[agent_handle])
            
        policy_shape_reward_timer.end()
        
        if render:
            env_renderer.render_env(
                    show=True, show_observations = False, frames = True, episode = True, step = True
                )
        
        # Update replay buffer and train agent
        for agent_handle in env.get_agent_handles():
            if update_values[agent_handle] or dones['__all__'] or deadlocked_agents[
                agent_handle]:
                # Only learn from timesteps where somethings happened
                policy_step_timer.start()
                policy.step(agent_handle,
                            agent_prev_obs[agent_handle],
                            agent_prev_action[agent_handle],
                            all_rewards[agent_handle],
                            agent_obs[agent_handle],
                            dones[agent_handle] or (deadlocked_agents[agent_handle] > 0))
                policy_step_timer.end()

                agent_prev_obs[agent_handle] = agent_obs[agent_handle].copy()
                agent_prev_action[agent_handle] = action_dict[agent_handle]

            score += all_rewards[agent_handle]

            # update_observation (step)
            agent_obs[agent_handle] = normalize_observation(next_obs[agent_handle], observation_tree_depth, observation_radius=observation_radius)

        nb_steps = step
        
        if dones['__all__']:
            break
        if deadlocked_agents['__all__']:  # deadlocked_agents['__has__']:
            if train_params.render_deadlocked is not None:
                # Setup renderer
                env_renderer = RenderTool(env,
                                            gl="PGL",
                                            show_debug=True,
                                            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS,
                                            screen_width=2000,
                                            screen_height=1200)

                env_renderer.set_new_rail()
                env_renderer.render_env(
                    show=False,
                    frames=True,
                    show_observations=False,
                    show_predictions=False
                )
                env_renderer.gl.save_image("{}/flatland_{:04d}.png".format(
                    train_params.render_deadlocked,
                    episode_idx))
                break
        if check_conflicts(env):
            
            num_of_conflict += 1
            
            break
            
    # policy.end_episode
    policy_end_episode_timer.start()
    policy.end_episode(train=True)
    policy_end_episode_timer.end()
    
    # Epsilon decay
    eps_start = max(eps_end, eps_decay * eps_start)
    
    total_episode_timer.end()
    
    # metric near to 1 is great result
    for agent_handle in env.get_agent_handles():
        metric += env.calculate_metric_single_agent(timetable, agent_handle)
    
    metric = metric/len(env.get_agent_handles())
    
    
    # Collect information about training
    tasks_finished = sum(dones[idx] for idx in env.get_agent_handles())
    tasks_deadlocked = sum(deadlocked_agents[idx] for idx in env.get_agent_handles())
    completion = tasks_finished / max(1, env.get_num_agents())
    deadlocked = tasks_deadlocked / max(1, env.get_num_agents())
    normalized_score = score / max(1, env.get_num_agents())
    action_probs = action_count / max(1, np.sum(action_count))
    
    avg_num_of_conflict = num_of_conflict / (episode_idx + 1)

    scores_window.append(normalized_score)
    completion_window.append(completion)
    deadlocked_window.append(deadlocked)
    smoothed_normalized_score = np.mean(scores_window)
    smoothed_completion = np.mean(completion_window)
    smoothed_deadlocked = np.mean(deadlocked_window)
    
    if train_params.render:
        env_renderer.close_window()
        
    # Print logs
    if episode_idx % checkpoint_interval == 0 and episode_idx > 0:
        policy.save('./checkpoints/' + training_id + '-' + str(episode_idx) + '.pth')

        if save_replay_buffer:
            policy.save_replay_buffer(
                './replay_buffers/' + training_id + '-' + str(episode_idx) + '.pkl')
            
        # reset action count
        action_count = [0] * get_flatland_full_action_size()
        
    print(
        '\r🚂 Episode {}'
        '\t 🏆 Score: {:.3f}'
        ' Avg: {:.3f}'
        '\t 💯 Done: {}%'
        ' Avg: {:.3f}%'
        '\t Num of conflicts: {}'
        ' Avg: {:.3f}'
        '\t 🎲 Epsilon: {:.3f} '
        '\t 🔀 Action Probs: {}'
        '\t Metric: {}'.format(
            episode_idx,
            score,
            smoothed_normalized_score,
            100 * completion,
            100 * smoothed_completion,
            num_of_conflict,
            avg_num_of_conflict,
            eps_start,
            format_action_prob(action_probs),
            metric
        ), end=" ")
    print()
    
    # Evaluate policy and log results at some interval
    if episode_idx % checkpoint_interval == 0 and n_eval_episodes > 0 and episode_idx > 0:
        scores, completions, nb_steps_eval = eval_policy(env,
                                                            Observer,
                                                            policy,
                                                            train_params,
                                                            obs_params)

        writer.add_scalar("evaluation/scores_min", np.min(scores), episode_idx)
        writer.add_scalar("evaluation/scores_max", np.max(scores), episode_idx)
        writer.add_scalar("evaluation/scores_mean", np.mean(scores), episode_idx)
        writer.add_scalar("evaluation/scores_std", np.std(scores), episode_idx)
        writer.add_histogram("evaluation/scores", np.array(scores), episode_idx)
        writer.add_scalar("evaluation/completions_min", np.min(completions), episode_idx)
        writer.add_scalar("evaluation/completions_max", np.max(completions), episode_idx)
        writer.add_scalar("evaluation/completions_mean", np.mean(completions), episode_idx)
        writer.add_scalar("evaluation/completions_std", np.std(completions), episode_idx)
        writer.add_histogram("evaluation/completions", np.array(completions), episode_idx)
        writer.add_scalar("evaluation/nb_steps_min", np.min(nb_steps_eval), episode_idx)
        writer.add_scalar("evaluation/nb_steps_max", np.max(nb_steps_eval), episode_idx)
        writer.add_scalar("evaluation/nb_steps_mean", np.mean(nb_steps_eval), episode_idx)
        writer.add_scalar("evaluation/nb_steps_std", np.std(nb_steps_eval), episode_idx)
        writer.add_histogram("evaluation/nb_steps", np.array(nb_steps_eval), episode_idx)

        smoothing = 0.9
        smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(
            scores) * (
                                                    1.0 - smoothing)
        smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(
            completions) * (1.0 - smoothing)
        writer.add_scalar("evaluation/smoothed_score", smoothed_eval_normalized_score,
                            episode_idx)
        writer.add_scalar("evaluation/smoothed_completion", smoothed_eval_completion,
                            episode_idx)

    if episode_idx > 49:
        try:
            # Save logs to tensorboard
            writer.add_scalar("scene_done_training/completion_{}".format(env.n_agents),
                                np.mean(completion), episode_idx)
            writer.add_scalar("scene_dead_training/deadlocked_{}".format(env.n_agents),
                                np.mean(deadlocked), episode_idx)

            writer.add_scalar("training/score", normalized_score, episode_idx)
            writer.add_scalar("training/max_steps", max_steps, episode_idx)
            writer.add_scalar("training/smoothed_score", smoothed_normalized_score, episode_idx)
            writer.add_scalar("training/completion", np.mean(completion), episode_idx)
            writer.add_scalar("training/deadlocked", np.mean(deadlocked), episode_idx)
            writer.add_scalar("training/smoothed_completion", np.mean(smoothed_completion),
                                episode_idx)
            writer.add_scalar("training/smoothed_deadlocked", np.mean(smoothed_deadlocked),
                                episode_idx)
            writer.add_scalar("training/nb_steps", nb_steps, episode_idx)
            writer.add_scalar("training/n_agents", env.n_agents, episode_idx)
            writer.add_histogram("actions/distribution", np.array(actions_taken), episode_idx)
            writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING],
                                episode_idx)
            writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode_idx)
            writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD],
                                episode_idx)
            writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode_idx)
            writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode_idx)
            writer.add_scalar("training/epsilon", eps_start, episode_idx)
            writer.add_scalar("training/buffer_size", len(policy.memory), episode_idx)
            writer.add_scalar("training/loss", policy.loss, episode_idx)

            writer.add_scalar("timer/00_reset", reset_timer.get(), episode_idx)
            writer.add_scalar("timer/01_policy_start_episode", policy_start_episode_timer.get(),
                                episode_idx)
            writer.add_scalar("timer/02_policy_start_step", policy_start_step_timer.get(),
                                episode_idx)
            writer.add_scalar("timer/03_policy_act", policy_act_timer.get(), episode_idx)
            writer.add_scalar("timer/04_env_step", env_step_timer.get(), episode_idx)
            writer.add_scalar("timer/05_policy_shape_reward", policy_shape_reward_timer.get(),
                                episode_idx)
            writer.add_scalar("timer/06_policy_step", policy_step_timer.get(), episode_idx)
            writer.add_scalar("timer/07_policy_end_step", policy_end_step_timer.get(), episode_idx)
            writer.add_scalar("timer/08_policy_end_episode", policy_end_episode_timer.get(),
                                episode_idx)
            writer.add_scalar("timer/09_total_episode", total_episode_timer.get_current(),
                                episode_idx)
            writer.add_scalar("timer/10_total", training_timer.get_current(), episode_idx)
        except:
            print("ERROR in writer", actions_taken)

    writer.flush()