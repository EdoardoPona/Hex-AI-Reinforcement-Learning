"""
Hex Reinforcement Learning Environment 
"""

import pygame
import random
import math
import numpy as np
import copy
import torch
import time

pi = 3.1415926535
red = (240, 0, 0)
blue = (0, 0, 240)
gray = (100, 100, 100)

colors = [red, blue, gray]


def distance(p0, p1):
    return math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)


class Hexagon:
    """ GUI hexagon for hex """
    def __init__(self, i, r=20):
        self.r = r
        # center coordinates
        yi = i % 8
        self.y = yi * (r + r*math.sin(pi/6) + 3) + 200

        self.x = i % 8 * (r * math.cos(pi / 6) + 3) + i/8 * (r*math.cos(pi/6)+3)*2 + 200

        self.points = ((self.x, self.y+r), (self.x-r*math.cos(pi/6), self.y+r*math.sin(pi/6)),
                       (self.x-r*math.cos(pi/6), self.y-r*math.sin(pi/6)), (self.x, self.y-r),
                       (self.x+r*math.cos(pi/6), self.y-r*math.sin(pi/6)), (self.x+r*math.cos(pi/6), self.y+r*math.sin(pi/6)))

    def render(self, display, color):

        # pygame.draw.circle(display, self.color, (int(self.x), int(self.y)), self.r)
        pygame.draw.polygon(display, color, self.points)


class Hex:

    """ Hex game environment for RL """
    def __init__(self, is_training=False):
        """ Red: 0, top-bottom; Blue: 1, right-left """
        super(Hex, self).__init__()
        pygame.init()

        self.is_training = is_training
        if is_training:
            self.FPS = 1000000
        else:
            self.FPS = 60
        self.display = pygame.display.set_mode((800, 700))
        self.clock = pygame.time.Clock()

        self.hexes = [Hexagon(i) for i in range(64)]

        self.turn = 0   # red

        self.paths = [[], []]   # lists of paths for each player
        # each path is of the form:  {'indices': [connected indices], 'right_connection_num': n...}
        # paths in paths[0] will have 'top_connection' and 'bottom_connection'
        self.available = list(range(64))

        self.state = torch.zeros(64) -1
        self.win_state = None

        # this dictionary is useful to check for edge connections, 'egde_keys' contains the key to be used to find the
        # corresponding attributes in self.paths[self.turn][i] objects
        # self.edges['attr'][self.turn] limits the responses to those relevant in the current turn
        # (no top data will be given in blue's turn for example)
        # 'edge_indices' contains the top, bottom, left, right edge indices to check wether a path is connected to a certain edge
        self.edges = {'edge_keys': [['top_c_num', 'bottom_c_num'], ['left_c_num', 'right_c_num']],
                      'edge_indices': [[[i*8 for i in range(8)], [(i+1)*8-1 for i in range(8)]],
                                       [list(range(8)), list(range(55, 64))]]}


    def step(self, index):

        if self.state[index] == -1:
            self.connect(index)

            if self.win_state is not None:
                reward = 20
                done = True

            else:
                reward = -1
                done = False

            self.turn = 1 if self.turn == 0 else 0
            return self.state, reward, done
        else:           # this should be avoided by filtering he actions
            print('invalid action')
            return self.state, -10, True

    def connect(self, index):
        # this hex is no longer available
        del self.available[self.available.index(index)]

        adj = [index - 1, index + 1, index - 7, index - 8, index + 7, index + 8]        # connectible hexes

        # checking for edge connections to add to the paths and to check for winners
        new_edge_connection = None
        for j in range(2):
            if index in self.edges['edge_indices'][self.turn][j]:
                new_edge_connection = self.edges['edge_keys'][self.turn][j]


        # considering edge cases
        if index % 8 == 0:
            del adj[adj.index(index - 1)]
            del adj[adj.index(index + 7)]
        elif index % 8 == 7:
            del adj[adj.index(index + 1)]
            del adj[adj.index(index - 7)]
        if index >= 56:
            try:
                del adj[adj.index(index + 7)]
            except:
                pass
            del adj[adj.index(index + 8)]
        elif index <= 7:
            try:
                del adj[adj.index(index - 7)]
            except:
                pass
            del adj[adj.index(index - 8)]

        where_to_add = []           # what paths is the node connected to?
        self.state[index] = self.turn
        for i, path in enumerate(self.paths[self.turn]):
            path_indices = path['indices']
            added_to_current_path = False

            for node in path_indices:
                for a in adj:
                    if node == a:
                        where_to_add.append(i)
                        added_to_current_path = True
                        break
                if added_to_current_path:
                    break

        if len(where_to_add) > 1:    # there is more than one path connected to this new edge, they should be merged
            merged_path = {'indices': [],
                           self.edges['edge_keys'][self.turn][0]: 0, self.edges['edge_keys'][self.turn][1]: 0}

            for i in where_to_add:
                merged_path['indices'] += copy.deepcopy(self.paths[self.turn][i]['indices'])

                for j in range(2):
                    edge_key = self.edges['edge_keys'][self.turn][j]
                    merged_path[edge_key] += self.paths[self.turn][i][edge_key]         # adding the number of connections to edges for all the paths


            merged_path['indices'].append(index)
            if new_edge_connection is not None:
                merged_path[new_edge_connection] += 1

            self.paths[self.turn].append(merged_path)

            for i, j in enumerate(where_to_add):
                del self.paths[self.turn][j-i]      # when you remove items, you change the other's indices, -i accounts for that

        elif len(where_to_add) == 1:
            self.paths[self.turn][where_to_add[0]]['indices'].append(index)
            if new_edge_connection is not None:
                self.paths[self.turn][where_to_add[0]][new_edge_connection] += 1

        else:
            new_path = {'indices': [index],
                        self.edges['edge_keys'][self.turn][0]: 0, self.edges['edge_keys'][self.turn][1]: 0}

            if new_edge_connection is not None:
                new_path[new_edge_connection] += 1

            self.paths[self.turn].append(new_path)

        for path in self.paths[self.turn]:
            if path[self.edges['edge_keys'][self.turn][0]] > 0 and path[self.edges['edge_keys'][self.turn][1]] > 0:
                self.win_state = self.turn

    def get_board(self):
        return self.state.view(1, 8, 8).float()

    def reset(self):
        self.turn = 0  # red
        self.paths = [[], []]  # lists of paths for each player
        # each path is of the form:  {'indices': [connected indices], 'right_connection_num': n...}
        # paths in paths[0] will have 'top_connection' and 'bottom_connection'
        self.available = list(range(64))
        self.state = torch.zeros(64) - 1
        self.win_state = None
        return self.state

    def render(self):
        self.display.fill((255, 255, 255))
        for i, hex in enumerate(self.hexes):
            hex.render(self.display, colors[int(self.state[i])])

        pygame.display.update()
        self.clock.tick(self.FPS)


def play_hex():
    """ you can play hex on your own or with a friend, or adapt this to play against a trained AI agent """"
    game = Hex()

    play = True
    mouse_pressed = False
    while play:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
                pygame.quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.reset()

        if pygame.mouse.get_pressed()[0] and not mouse_pressed:
            pos = pygame.mouse.get_pos()
            distances = [distance(pos, (h.x, h.y)) for h in game.hexes]

            index = distances.index(min(distances))

            state, rewards, done = game.step(index)
            mouse_pressed = True
        elif not pygame.mouse.get_pressed()[0]:
            mouse_pressed = False

        game.render()

