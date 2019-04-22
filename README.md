# Hex-AI-Reinforcement-Learning
Reinforcement Learning agents for the game of Hex

Currently the only algorithm implemented is REINFORCE. There are two ways to train, against a player that picks random moves
or via self play. When self playing the policy takes as input a turn_vector as well so to decide on whose side it should play. Otherwise when playing against a random player, at the moment the AI is always the red player.
Both methods are very slow at learning. To remedy this I will add a baseline. 

Coming soon: 
Baseline
Actor-Critic
Proximal Policy Optimization
