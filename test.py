from asyncio import Future
import asyncio
from asyncio.queues import Queue
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

import tensorflow as tf
import numpy as np
import os
import sys
import random
import time
import argparse
from collections import deque, defaultdict, namedtuple
import copy
from policy_value_network_tf2 import *
from policy_value_network_gpus_tf2 import *
import scipy.stats
from threading import Lock
from concurrent.futures import ThreadPoolExecutor


class Move():
    def __init__(self, from_x, from_y, to_x, to_y):
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y

    def __eq__(self, other):
        return self.from_x == other.from_x and self.from_y == other.from_y and self.to_x == other.to_x and self.to_y == other.to_y


class Chess():
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y


QueueItem = namedtuple("QueueItem", "feature future")
c_PUCT = 5
virtual_loss = 3
cut_off_depth = 30


class leaf_node(object):
    def __init__(self, in_parent, in_prior_p, board):
        self.P = in_prior_p
        self.Q = 0
        self.N = 0
        self.v = 0
        self.U = 0
        self.W = 0
        self.parent = in_parent
        self.child = {}
        self.board = board

    def is_leaf(self):
        return self.child == {}

    def get_Q_plus_U_new(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        # self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        U = c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + U

    def get_Q_plus_U(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        # self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        self.U = c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + self.U

    # def select_move_by_action_score(self, noise=True):
    #
    #     # P = params[self.lookup['P']]
    #     # N = params[self.lookup['N']]
    #     # Q = params[self.lookup['W']] / (N + 1e-8)
    #     # U = c_PUCT * P * np.sqrt(np.sum(N)) / (1 + N)
    #
    #     ret_a = None
    #     ret_n = None
    #     action_idx = {}
    #     action_score = []
    #     i = 0
    #     for a, n in self.child.items():
    #         U = c_PUCT * n.P * np.sqrt(n.parent.N) / ( 1 + n.N)
    #         action_idx[i] = (a, n)
    #
    #         if noise:
    #             action_score.append(n.Q + U * (0.75 * n.P + 0.25 * dirichlet([.03] * (go.N ** 2 + 1))) / (n.P + 1e-8))
    #         else:
    #             action_score.append(n.Q + U)
    #         i += 1
    #         # if(n.Q + n.U > max_Q_plus_U):
    #         #     max_Q_plus_U = n.Q + n.U
    #         #     ret_a = a
    #         #     ret_n = n
    #
    #     action_t = int(np.argmax(action_score[:-1]))
    #
    #     return ret_a, ret_n
    #     # return action_t
    def select_new(self, c_puct):
        return max(self.child.items(), key=lambda node: node[1].get_Q_plus_U_new(c_puct))

    def select(self, c_puct):
        # max_Q_plus_U = 1e-10
        # ret_a = None
        # ret_n = None
        # for a, n in self.child.items():
        #     n.U = c_puct * n.P * np.sqrt(n.parent.N) / ( 1 + n.N)
        #     if(n.Q + n.U > max_Q_plus_U):
        #         max_Q_plus_U = n.Q + n.U
        #         ret_a = a
        #         ret_n = n
        # return ret_a, ret_n
        return max(self.child.items(), key=lambda node: node[1].get_Q_plus_U(c_puct))

    # @profile
    def expand(self, moves, action_probs):
        tot_p = 1e-8
        # print("action_probs : ", action_probs)
        action_probs = tf.squeeze(action_probs)  # .flatten()   #.squeeze()
        # print("expand action_probs shape : ", action_probs.shape)
        for action in moves:
            board = GameBoard.make_move(action, self.board)
            mov_p = action_probs[label2i[action]]
            new_node = leaf_node(self, mov_p, board)
            self.child[action] = new_node
            tot_p += mov_p

        for a, n in self.child.items():
            n.P /= tot_p

    def back_up_value(self, value):
        self.N += 1
        self.W += value
        self.v = value
        self.Q = self.W / self.N  # node.Q += 1.0*(value - node.Q) / node.N
        self.U = c_PUCT * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        # node = node.parent
        # value = -value

    def backup(self, value):
        node = self
        while node != None:
            node.N += 1
            node.W += value
            node.v = value
            node.Q = node.W / node.N  # node.Q += 1.0*(value - node.Q) / node.N
            node = node.parent
            value = -value


class MCTS_tree(object):
    def __init__(self, board, in_forward, search_threads):
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3  # 0.03
        self.p_ = (1 - self.noise_eps) * 1 + self.noise_eps * np.random.dirichlet([self.dirichlet_alpha])
        self.root = leaf_node(None, self.p_, board)
        self.c_puct = 5  # 1.5
        # self.policy_network = in_policy_network
        self.forward = in_forward
        self.node_lock = defaultdict(Lock)

        self.virtual_loss = 3
        self.now_expanding = set()
        self.expanded = set()
        self.cut_off_depth = 30
        # self.QueueItem = namedtuple("QueueItem", "feature future")
        self.sem = asyncio.Semaphore(search_threads)
        self.queue = Queue(search_threads)
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0

    def reload(self):
        self.root - leaf_node(None, self.p_, '-1-1-1-1-1-1/-1-1-1-1-1-1/6/6/111111/111111')
        '''self.root = leaf_node(None, self.p_,
                              "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr")  # "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"'''
        self.expanded = set()

    def Q(self, move) -> float:  # type hint, Q() returns float
        ret = 0.0
        find = False
        for a, n in self.root.child.items():
            if move == a:
                ret = n.Q
                find = True
        if (find == False):
            print("{} not exist in the child".format(move))
        return ret

    def update_tree(self, act):
        # if(act in self.root.child):
        self.expanded.discard(self.root)
        self.root = self.root.child[act]
        self.root.parent = None

    def is_expanded(self, key) -> bool:
        """Check expanded status"""
        return key in self.expanded

    async def tree_search(self, node, current_player, restrict_round) -> float:
        """Independent MCTS, stands for one simulation"""
        self.running_simulation_num += 1

        # reduce parallel search number
        with await self.sem:
            value = await self.start_tree_search(node, current_player, restrict_round)
            # logger.debug(f"value: {value}")
            # logger.debug(f'Current running threads : {RUNNING_SIMULATION_NUM}')
            self.running_simulation_num -= 1

            return value

    async def start_tree_search(self, node, current_player, restrict_round) -> float:
        """Monte Carlo Tree search Select,Expand,Evauate,Backup"""
        now_expanding = self.now_expanding

        while node in now_expanding:
            await asyncio.sleep(1e-4)

        if not self.is_expanded(node):  # and node.is_leaf()
            """is leaf node try evaluate and expand"""
            # add leaf node to expanding list
            self.now_expanding.add(node)

            positions = self.generate_inputs(node.board, current_player)
            # positions = np.expand_dims(positions, 0)

            # push extracted dihedral features of leaf node to the evaluation queue
            future = await self.push_queue(positions)  # type: Future
            await future
            action_probs, value = future.result()

            # action_probs, value = self.forward(positions)
            if self.is_black_turn(current_player):
                action_probs = cchess_main.flip_policy(action_probs)

            moves = GameBoard.move_generate(node.board, current_player)
            # print("current_player : ", current_player)
            # print(moves)
            node.expand(moves, action_probs)
            self.expanded.add(node)  # node.board

            # remove leaf node from expanding list
            self.now_expanding.remove(node)

            # must invert, because alternative layer has opposite objective
            return value[0] * -1

        else:
            """node has already expanded. Enter select phase."""
            # select child node with maximum action scroe
            last_board = node.board

            action, node = node.select_new(c_PUCT)
            current_player = "w" if current_player == "b" else "b"
            if is_kill_move(last_board, node.board) == 0:
                restrict_round += 1
            else:
                restrict_round = 0
            last_board = node.board

            # action_t = self.select_move_by_action_score(key, noise=True)

            # add virtual loss
            # self.virtual_loss_do(key, action_t)
            node.N += virtual_loss
            node.W += -virtual_loss

            # evolve game board status
            # child_position = self.env_action(position, action_t)

            if node.board.judge(current_player) != 0:

                value = 1.0 if node.board.judge(current_player) == 1 else -1.0
                value = -1.0 if node.board.judge(current_player) == -1 else 1.0
                value = value * -1

            elif restrict_round >= 60:
                value = 0.0
            else:
                value = await self.start_tree_search(node, current_player, restrict_round)  # next move
            # if node is not None:
            #     value = await self.start_tree_search(node)  # next move
            # else:
            #     # None position means illegal move
            #     value = -1

            # self.virtual_loss_undo(key, action_t)
            node.N += -virtual_loss
            node.W += virtual_loss

            # on returning search path
            # update: N, W, Q, U
            # self.back_up_value(key, action_t, value)
            node.back_up_value(value)  # -value

            # must invert
            return value * -1
            # if child_position is not None:
            #     return value * -1
            # else:
            #     # illegal move doesn't mean much for the opponent
            #     return 0

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        """
        q = self.queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(1e-3)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            # logger.debug(f"predicting {len(item_list)} items")
            features = np.asarray([item.feature for item in item_list])  # asarray
            # print("prediction_worker [features.shape] before : ", features.shape)
            # shape = features.shape
            # features = features.reshape((shape[0] * shape[1], shape[2], shape[3], shape[4]))
            # print("prediction_worker [features.shape] after : ", features.shape)
            # policy_ary, value_ary = self.run_many(features)
            action_probs, value = self.forward(features)
            for p, v, item in zip(action_probs, value, item_list):
                item.future.set_result((p, v))

    async def push_queue(self, features):
        future = self.loop.create_future()
        item = QueueItem(features, future)
        await self.queue.put(item)
        return future

    # @profile
    def main(self, board, current_player, restrict_round, playouts):
        node = self.root
        if not self.is_expanded(node):  # and node.is_leaf()    # node.board
            # print('Expadning Root Node...')
            positions = self.generate_inputs(node.board, current_player)
            positions = np.expand_dims(positions, 0)
            action_probs, value = self.forward(positions)
            if self.is_black_turn(current_player):
                action_probs = cchess_main.flip_policy(action_probs)

            moves = GameBoard.move_generate(node.board, current_player)
            # print("current_player : ", current_player)
            # print(moves)
            node.expand(moves, action_probs)
            self.expanded.add(node)  # node.board

        coroutine_list = []
        for _ in range(playouts):
            coroutine_list.append(self.tree_search(node, current_player, restrict_round))
        coroutine_list.append(self.prediction_worker())
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))

    def do_simulation(self, board, current_player, restrict_round):
        node = self.root
        last_board = board
        while (node.is_leaf() == False):
            # print("do_simulation while current_player : ", current_player)
            action, node = node.select(self.c_puct)
            current_player = "w" if current_player == "b" else "b"
            if is_kill_move(last_board, node.board) == 0:
                restrict_round += 1
            else:
                restrict_round = 0
            last_board = node.board

        positions = self.generate_inputs(node.board, current_player)
        positions = np.expand_dims(positions, 0)
        action_probs, value = self.forward(positions)
        if self.is_black_turn(current_player):
            action_probs = cchess_main.flip_policy(action_probs)

        # print("action_probs shape : ", action_probs.shape)    #(1, 2086)

        if node.board.judge(current_player) != 0:

            value = 1.0 if node.board.judge(current_player) == 1 else -1.0
            value = -1.0 if node.board.judge(current_player) == -1 else 1.0
            value = value * -1
        elif restrict_round >= 60:
            value = 0.0
        else:
            moves = GameBoard.move_generate(node.board, current_player)
            # print("current_player : ", current_player)
            # print(moves)
            node.expand(moves, action_probs)

        node.backup(-value)

    def generate_inputs(self, board, current_player):
        board, palyer = self.try_flip(board, current_player, self.is_black_turn(current_player))
        return self.state_to_positions(board)

    def replace_board_tags(self, board):
        board = board.replace("2", "11")
        board = board.replace("3", "111")
        board = board.replace("4", "1111")
        board = board.replace("5", "11111")
        board = board.replace("6", "111111")
        board = board.replace("7", "1111111")
        board = board.replace("8", "11111111")
        board = board.replace("9", "111111111")
        return board.replace("/", "")

    # 感觉位置有点反了，当前角色的棋子在右侧，plane的后面
    def state_to_positions(self, board):
        # TODO C plain x 2
        board_state = self.replace_board_tags(state)
        pieces_plane = np.zeros(shape=(9, 10, 14), dtype=np.float32)
        for rank in range(9):  # 横线
            for file in range(10):  # 直线
                v = board_state[rank * 9 + file]
                if v.isalpha():
                    pieces_plane[rank][file][ind[v]] = 1
        assert pieces_plane.shape == (9, 10, 14)
        return pieces_plane

    def try_flip(self, board, current_player, flip=False):
        if not flip:
            return board, current_player

        rows = [''.join(i) for i in board]                  #output rows in format ['000000', '111111', '222222', '333333', '444444', '555555']

        def swapcase(a):
            if a.isalpha():
                return a.lower() if a.isupper() else a.upper()
            return a

        def swapall(aa):
            return "".join([swapcase(a) for a in aa])

        return "/".join([swapall(row) for row in reversed(rows)]), ('w' if current_player == 'b' else 'b')

    def is_black_turn(self, current_player):
        return current_player == 'b'


class GameBoard(object):
    board = [[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    blackNum = 6
    whiteNum = 6
    state = '-1-1-1-1-1-1/-1-1-1-1-1-1/6/6/111111/111111'

    def __init__(self):
        self.round = 1
        self.player = 1
        self.restrict_round = 0

    def __init__(self, board):
        self.board = board
        self.round = 1
        self.player = 1
        self.restrict_round = 0

    @staticmethod
    def print_board(board):
        for i in range(6):
            for j in range(6):
                print(board[i][j], " ", end="")
            print()
        '''
        for i in range(6):
            total = ' '.join(board[i])
            print(total)
        '''


    def judge(self, currentplayer):
        if currentplayer == -1:  # blackchess -1
            if self.blackNum == 0:
                return -1
            elif self.whiteNum == 0:
                return 1
            else:
                return 0
        elif currentplayer == 1:  # whitechess 1
            if self.blackNum == 0:
                return 1
            elif self.whiteNum == 0:
                return -1
            else:
                return 0

    @staticmethod
    def move_generate(game_board, current_player):
        moves = []
        exrool_s = []
        exrool_chess_s = []
        inrool_s = []
        inrool_chess_s = []

        inside_rool = 1
        exterior_rool = 2
        inrool_s, inrool_chess_s = game_board.extract_rool(game_board, inside_rool)
        exrool_s, exrool_chess_s = game_board.extract_rool(game_board, exterior_rool)

        game_board.attack_generate(moves, exrool_s, exrool_chess_s, current_player)
        game_board.attack_generate(moves, inrool_s, inrool_chess_s, current_player)

        for i in range(6):
            for j in range(6):
                if game_board.board[i][j] is current_player:
                    for k in range(i - 1, i + 2):
                        for l in range(j - 1, j + 2):
                            if 0 <= k <= 5:
                                if 0 <= l <= 5:
                                    if i is not k or j is not l:
                                        if game_board.board[k][l] is 0:
                                            moves.append(Move(i, j, k, l))
                                            # print(i,j,k,l)

        for _ in range(len(exrool_s)):
            try:
                exrool_s.remove([0, 0, 0, 0, 0, 0])
            except:
                pass

        return moves

    @staticmethod
    def extract_rool(game_board, index):
        exrool_s = []
        exrool_chess_s = []
        exrool = []
        exrool_chess = []
        inrool_s = []
        exrool_chess_s = []
        inrool = []
        inrool_chess = []

        for i in range(6):
            exrool.append(game_board.board[index][i])
            exrool_chess.append(Chess(game_board.board[index][i], index, i))
        exrool_s.append(exrool)
        exrool_chess_s.append(exrool_chess)

        exrool = []
        exrool_chess = []
        for i in range(6):
            exrool.append(game_board.board[i][5 - index])
            exrool_chess.append(Chess(game_board.board[i][5 - index], i, 5 - index))
        exrool_s.append(exrool)
        exrool_chess_s.append(exrool_chess)

        exrool = []
        exrool_chess = []
        for i in reversed(range(6)):
            exrool.append(game_board.board[5 - index][i])
            exrool_chess.append(Chess(game_board.board[5 - index][i], 5 - index, i))
        exrool_s.append(exrool)
        exrool_chess_s.append(exrool_chess)

        exrool = []
        exrool_chess = []
        for i in reversed(range(6)):
            exrool.append(game_board.board[i][index])
            exrool_chess.append(Chess(game_board.board[i][index], i, index))
        exrool_s.append(exrool)
        exrool_chess_s.append(exrool_chess)

        for _ in range(len(exrool_s)):
            try:
                exrool_chess_s.pop(exrool_s.index([0, 0, 0, 0, 0, 0]))
                exrool_s.remove([0, 0, 0, 0, 0, 0])
            except:
                pass

        return exrool_s, exrool_chess_s

    @staticmethod
    def attack_generate(moves, rools, rools_chess, who):

        for i in range(len(rools)):
            p = -1
            for j in reversed(range(6)):
                if rools[i][j] is who:
                    p = j
                    break
                elif rools[i][j] is -who:
                    break
            if p < 0:
                continue

            p_n = -1
            for j in range(6):
                if rools[(i + 1) % (len(rools))][j] is -who:
                    p_n = j
                    break
                elif rools[(i + 1) % (len(rools))][j] is who:
                    break

            if p_n is -1:
                try:
                    p_n = rools[(i + 1) % (len(rools))].index(who)
                    if rools_chess[i][p].x is rools_chess[(i + 1) % (len(rools))][p_n].x and rools_chess[i][p].y is \
                            rools_chess[(i + 1) % (len(rools))][
                                p_n].y:
                        flag = False
                        for j in range(p_n + 1, 6):
                            if rools[(i + 1) % (len(rools))][j] is who:
                                break
                            elif rools[(i + 1) % (len(rools))][j] is -who:
                                p_n = j
                                flag = True
                                break
                        if flag is True:
                            if Move(rools_chess[i][p].x, rools_chess[i][p].y,
                                    rools_chess[(i + 1) % (len(rools))][p_n].x,
                                    rools_chess[(i + 1) % (len(rools))][p_n].y) not in moves:
                                moves.append(Move(rools_chess[i][p].x, rools_chess[i][p].y,
                                                  rools_chess[(i + 1) % (len(rools))][p_n].x,
                                                  rools_chess[(i + 1) % (len(rools))][p_n].y))
                except:
                    pass

            else:
                if Move(rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[(i + 1) % (len(rools))][p_n].x,
                        rools_chess[(i + 1) % (len(rools))][p_n].y) not in moves:
                    moves.append(
                        Move(rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[(i + 1) % (len(rools))][p_n].x,
                             rools_chess[(i + 1) % (len(rools))][p_n].y))

        for i in range(len(rools)):
            p = -1
            for j in range(6):
                if rools[i][j] is who:
                    p = j
                    break
                elif rools[i][j] is -who:
                    break
            if p < 0:
                continue

            p_n = -1
            for j in reversed(range(6)):
                if rools[i - 1][j] is -who:
                    p_n = j
                    break
                elif rools[i - 1][j] is who:
                    break

            if p_n is -1:
                try:
                    for j in reversed(range(6)):
                        if rools[i - 1][j] is who:
                            p_n = j
                            break
                    if rools_chess[i][p].x is rools_chess[i - 1][p_n].x and rools_chess[i][p].y is \
                            rools_chess[i - 1][
                                p_n].y:
                        flag = False
                        for j in reversed(range(0, p_n)):
                            if rools[i - 1][j] is who:
                                break
                            elif rools[i - 1][j] is -who:
                                p_n = j
                                flag = True
                                break
                        if flag is True:
                            if Move(rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[i - 1][p_n].x,
                                    rools_chess[i - 1][p_n].y) not in moves:
                                moves.append(
                                    Move(rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[i - 1][p_n].x,
                                         rools_chess[i - 1][p_n].y))
                except:
                    pass
            else:

                if Move(rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[i - 1][p_n].x,
                        rools_chess[i - 1][p_n].y) not in moves:
                    moves.append(Move(rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[i - 1][p_n].x,
                                      rools_chess[i - 1][p_n].y))

        return moves

    @staticmethod
    def make_move(self, in_action, board):
        return 0

    def softmax(x):
        # print(x)
        probs = np.exp(x - np.max(x))
        # print(np.sum(probs))
        probs /= np.sum(probs)
        return probs


class surakarta(object):
    def __init__(self, playout=400, in_batch_size=128, exploration=True, in_search_threads=16, processor="cpu",
                 num_gpus=1, res_block_nums=7, human_color='b'):
        self.epochs = 5
        self.playout_counts = playout  # 400    #800    #1600    200
        self.temperature = 1  # 1e-8    1e-3
        # self.c = 1e-4
        self.batch_size = in_batch_size  # 128    #512
        # self.momentum = 0.9
        self.game_batch = 400  # Evaluation each 400 times
        # self.game_loop = 25000
        self.top_steps = 30
        self.top_temperature = 1  # 2
        # self.Dirichlet = 0.3    # P(s,a) = (1 - ϵ)p_a  + ϵη_a    #self-play chapter in the paper
        self.eta = 0.03
        # self.epsilon = 0.25
        # self.v_resign = 0.05
        # self.c_puct = 5
        self.learning_rate = 0.001  # 5e-3    #    0.001
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.game_borad = GameBoard()
        self.processor = processor
        # self.current_player = 'w'    #“w”表示红方，“b”表示黑方。
        self.policy_value_netowrk = policy_value_network(self.lr_callback,
                                                         res_block_nums) if processor == 'cpu' else policy_value_network_gpus(
            num_gpus, res_block_nums)
        self.search_threads = in_search_threads
        self.mcts = MCTS_tree(self.game_borad.state, self.policy_value_netowrk.forward, self.search_threads)
        self.exploration = exploration
        self.resign_threshold = -0.8  # 0.05
        self.global_step = 0
        self.kl_targ = 0.025
        self.log_file = open(os.path.join(os.getcwd(), 'log_file.txt'), 'w')
        self.human_color = human_color


# gameBoard = GameBoard()
# gameBoard.print_board(gameBoard.board)
# gameBoard.move_generate(gameBoard, -1)

if __name__ == '__main__':
    file = open("testMoveGenerate.txt")
    k = int(input('How much tests do you want to take: '))

    start = time.time()
    t_board = [[]] * 6
    '''board[0] = [1,1,1,1,1,1]
    print(board)'''
    for tests in range(k):
        for i in range(6):
            t_board[i] = list(map(int, file.readline().split()))
        meta = list(map(int, file.readline().split()))
        for i in range(6):
            for j in range(6):
                if t_board[i][j] is 2:
                    t_board[i][j] = -1
        gameboard = GameBoard(t_board)
        move_b = gameboard.move_generate(gameboard, -1)
        move_w = gameboard.move_generate(gameboard, 1)
        assert len(move_b) == meta[2]
        assert len(move_w) == meta[3]

    print('Run test for move generator in', time.time() - start)
