import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from torch.distributions import Categorical
import Hex
import Models
import time
import copy
import random

env = Hex.Hex()
policy = Models.PolicyTwoSides().cuda()

optimizer = torch.optim.Adam(policy.parameters(), 1e-4)

gamma = 0.99


def generate_data(game_num):
    """ ideally we would store the gradients as well,
    at the moment all we can do with this data is supervised learning at the end
    you would play all the games again and after each state update with the stored
    reward. This isn't really much faster than doing everything in the same run,
    as it requires the policy to be computed twice (generating the data and training it)
    however it would allow for batching."""

    games = []  # list of recorded games 			[[(state, (red_reward, blue_reward)), ...], ...]
    for i in range(game_num):
        state = env.reset()
        done = False
        current_game = {'states': [], 'turns': [], 'log_probs': [], 'red_rewards': [], 'blue_rewards': []}
        rewards = []  # both red and blue, will later get converted to G
        while not done:
            current_game['states'].append(state)
            turn_vector = torch.zeros(1, 10) + env.turn

            probs = policy(state.unsqueeze(0), turn_vector)
            m = Categorical(probs)
            action = m.sample()

            current_game['log_probs'].append(m.log_prob(action))
            state, reward, done = env.step(action.item())

            rewards.append(reward)
            env.render()
            time.sleep(0.5)

        rG = 0
        bG = 0

        for r, b in rewards[::-1]:
            rG = r + gamma * rG
            bG = b + gamma * bG
            current_game['red_rewards'].insert(0, rG)
            current_game['blue_rewards'].insert(0, bG)

            games.append(current_game)
    return games


def play_two_player_game():
    state = env.reset()
    game_length = 0
    done = False

    reward0 = 0
    log_probs0 = []

    reward1 = 0
    log_probs1 = []

    while not done:
        if env.turn == 0:

            turn_vector = torch.zeros(10).cuda().float()
            probs = policy(state.view(1, 8, 8).unsqueeze(0).cuda(), env.available, turn_vector)

            m = Categorical(probs)
            action = m.sample()
            a = action.item()
            action_index = env.available[a]

            state, r, done = env.step(action_index)

            log_probs0.append(m.log_prob(action))
            reward0 += r

            if done:
                reward1 -= r

        else:

            turn_vector = torch.ones(10).cuda().float()
            probs = policy(state.view(1, 8, 8).unsqueeze(0).cuda(), env.available, turn_vector)

            m = Categorical(probs)
            action = m.sample()
            a = action.item()
            action_index = env.available[a]

            state, r, done = env.step(action_index)

            log_probs1.append(m.log_prob(action))
            reward1 += r

            if done:
                reward0 -= r

        game_length += 1

    return reward0, reward1, log_probs0, log_probs1, game_length, env.win_state


def play_and_train_two_sides(game_num):
    past_winners = []
    for g_num in range(game_num):
        r0, r1, log_prob0, log_prob1, game_length, winner = play_two_player_game()

        update_weights(r0, log_prob0, game_length)
        update_weights(r1, log_prob1, game_length)

        past_winners.append(winner)
        if len(past_winners) > 20:
            del past_winners[0]

        if g_num % 500 == 0:
            print('game', g_num, 'reward0', r0, 'reward1', r1, 'blue has won', sum(past_winners) / 20, 'of the last 20 games')

        if g_num % 1000 == 0:

            print('Red vs random', play_random(0, 20)/20, 'of 20 games')
            print('Blue vs random', play_random(1, 20)/20, 'of 20 games')


def play_random(player, game_num):
    wins = 0
    for i in range(game_num):
        state = env.reset()
        done = False
        while not done:
            if env.turn == player:
                turn_vector = torch.zeros(10).cuda() + player
                probs = policy(state.view(1, 8, 8).unsqueeze(0).cuda(), env.available, turn_vector)

                m = Categorical(probs)
                action = m.sample()
                a = action.item()
                action_index = env.available[a]

                state, r, done = env.step(action_index)

                if done:
                    wins += 1
            else:
                state, r, done = env.step(random.choice(env.available))

    return wins


def update_weights(R, log_probs, game_length):
    policy_rewards = [R for i in range(game_length)]
    G = 0
    policy_loss = []
    rewards = []

    for r in policy_rewards[::-1]:
        G = r + gamma*G
        rewards.insert(0, G)

    rewards = torch.Tensor(rewards).cuda()
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    for log_prob, reward in zip(log_probs, rewards):
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()


def play_a_game():
    state = env.reset()
    game_length = 0
    done = False
    reward = 0
    while not done:
        if env.turn == 0:
            probs = policy(state.view(1, 8, 8).unsqueeze(0).cuda(), env.available)

            m = Categorical(probs)
            action = m.sample()
            a = action.item()
            action_index = env.available[a]

            state, r, done = env.step(action_index)

            policy.log_probs.append(m.log_prob(action))
            reward += r

        else:
            state, r , done = env.step(random.choice(env.available))  # player 1, random move
            if done:
                reward += -r         # only adding penalty when loosing

        game_length += 1

    return reward, game_length, env.win_state

def play_and_train_one_side(game_num):
    past_winners = []
    for g_num in range(game_num):
        game_reward, game_length, game_winner = play_a_game()
        policy.rewards = [game_reward for i in range(game_length)]

        G = 0
        policy_loss = []
        rewards = []

        for r in policy.rewards[::-1]:
            G = r + gamma*G
            rewards.insert(0, G)

        rewards = torch.Tensor(rewards).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for log_prob, reward in zip(policy.log_probs, rewards):
            policy_loss.append(-log_prob*reward)

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        del policy.rewards[:]
        del policy.log_probs[:]

        past_winners.append(game_winner)
        if len(past_winners) > 20:
            del past_winners[0]

        if g_num % 500 == 0:
            print('game number', g_num, ', rewards', game_reward, ', game length', game_length,
                  ', win percentage', (20-sum(past_winners))/len(past_winners))


play_and_train_two_sides(int(1e6))


# TODO
# Actor-Critic algorithm
# Proximal Policy Optimization
# board position heuristics
# MCTS or minimax player for training
