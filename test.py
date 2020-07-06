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
# from policy_value_network_tf2 import *
# from policy_value_network_gpus_tf2 import *
import scipy.stats
from threading import Lock
from concurrent.futures import ThreadPoolExecutor



def create_uci_labels():
    labels_array = []

    for i in range(6):
        for j in range(6):
            for k in range(6):
                for l in range(6):
                    if bool(i is not k) or bool(j is not l):
                        labels_array.append(str(i) + str(j) +str(k) + str(l))

    return labels_array

def is_kill_move(state_prev, state_next):
    return state_next.blackNum - state_prev.blackNum + state_next.whiteNum - state_prev.whiteNum

labels_array = create_uci_labels()
labels_len = len(labels_array)
label2i = {val: i for i, val in enumerate(labels_array)}

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
        u = c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + u

    def get_Q_plus_U(self, c_puct):
        self.U = c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + self.U

    def select_new(self, c_puct):
        return max(self.child.items(), key=lambda node: node[1].get_Q_plus_U_new(c_puct))

    def select(self, c_puct):
        return max(self.child.items(), key=lambda node: node[1].get_Q_plus_U(c_puct))

    def expand(self, moves, action_probs):
        tot_p = 1e-8
        action_probs = tf.squeeze(action_probs)
        for action in moves:
            board = GameBoard.make_move(action, self.board[:])
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
        self.Q = self.W / self.N
        self.U = c_PUCT * self.P * np.sqrt(self.parent.N) / (1 + self.N)

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
        self.root = leaf_node(None, self.p_, GameBoard().board)
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
        return key in self.expanded

    async def tree_search(self, node, current_player, restrict_round) -> float:
        self.running_simulation_num += 1

        with await self.sem:
            value = await self.start_tree_search(node, current_player, restrict_round)
            self.running_simulation_num -= 1

            return value

    async def start_tree_search(self, node, current_player, restrict_round) -> float:
        now_expanding = self.now_expanding

        while node in now_expanding:
            await asyncio.sleep(1e-4)

        if not self.is_expanded(node):
            self.now_expanding.add(node)

            positions = self.generate_inputs(node.board, current_player)

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
        board, player = self.try_flip(board, current_player, self.is_black_turn(current_player))
        return self.state_to_positions(board)



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

    def make_move(self, move, board):
        origin = board[move.to_x][move.to_y]
        player = board[move.from_x][move.from_x]
        board[move.from_x][move.from_y] = 0
        board[move.to_x][move.to_y] = player
        if origin is -1:
            self.blackNum = self.blackNum - 1
        elif origin is 1:
            self.whiteNum = self.whiteNum - 1
        return board

    def is_game_over(self):
        if self.blackNum is 0:
            return -1
        elif self.whiteNum is 0:
            return 1
        else:
            return False



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

    def run(self):
        batch_iter = 0
        try:
            while (True):
                batch_iter += 1
                play_data, episode_len = self.selfplay()
                print("batch i:{}, episode_len:{}".format(batch_iter, episode_len))
                extend_data = []
                # states_data = []
                for state, mcts_prob, winner in play_data:
                    states_data = self.mcts.state_to_positions(state)
                    # prob = np.zeros(labels_len)
                    # for idx in range(len(mcts_prob[0][0])):
                    #     prob[label2i[mcts_prob[0][0][idx]]] = mcts_prob[0][1][idx]
                    extend_data.append((states_data, mcts_prob, winner))
                self.data_buffer.extend(extend_data)
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
                # if (batch_iter) % self.game_batch == 0:
                #     print("current self-play batch: {}".format(batch_iter))
                #     win_ratio = self.policy_evaluate()
        except KeyboardInterrupt:
            self.log_file.close()
            self.policy_value_netowrk.save(self.global_step)


    def get_action(self, board, temperature=1e-3):

        self.mcts.main(board, self.game_borad.current_player, self.game_borad.restrict_round, self.playout_counts)

        actions_visits = [(act, nod.N) for act, nod in self.mcts.root.child.items()]
        actions, visits = zip(*actions_visits)
        probs = softmax(1.0 / temperature * np.log(visits))
        move_probs = []
        move_probs.append([actions, probs])

        if (self.exploration):
            act = np.random.choice(actions, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
        else:
            act = np.random.choice(actions, p=probs)

        win_rate = self.mcts.Q(act)
        self.mcts.update_tree(act)

        return act, move_probs, win_rate

    def selfplay(self):
        self.game_borad.reload()
        states, mcts_probs, current_players = [], [], []
        z = None
        game_over = False
        winnner = ""
        start_time = time.time()
        while (not game_over):
            action, probs, win_rate = self.get_action(self.game_borad.state, self.temperature)
            states.append(self.game_borad.board)
            prob = np.zeros(labels_len)
            for idx in range(len(probs[0][0])):
                prob[label2i[probs[0][0][idx]]] = probs[0][1][idx]
            mcts_probs.append(prob)
            current_players.append(self.game_borad.current_player)

            last_state = self.game_borad
            self.game_borad.board = GameBoard.make_move(action, self.game_borad.board)
            self.game_borad.round += 1
            self.game_borad.current_player = -self.game_borad.current_player
            if is_kill_move(last_state, self.game_borad) == 0:
                self.game_borad.restrict_round += 1
            else:
                self.game_borad.restrict_round = 0

            # self.game_borad.print_borad(self.game_borad.state, action)

            if (self.game_borad.is_game_over() is not False):
                z = np.zeros(len(current_players))
                if (self.game_borad.is_game_over() == -1):
                    winnner = -1
                if (self.game_borad.is_game_over() == 1):
                    winnner = 1
                z[np.array(current_players) == winnner] = 1.0
                z[np.array(current_players) != winnner] = -1.0
                game_over = True
                print("Game end. Winner is player : ", winnner, " In {} steps".format(self.game_borad.round - 1))
            elif self.game_borad.restrict_round >= 60:
                z = np.zeros(len(current_players))
                game_over = True
                print("Game end. Tie in {} steps".format(self.game_borad.round - 1))
            if (game_over):
                self.mcts.reload()
        print("Using time {} s".format(time.time() - start_time))
        return zip(states, mcts_probs, z), len(z)


def softmax(x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

if __name__ == '__main__':
    '''
    file = open("testMoveGenerate.txt")
    k = int(input('How much tests do you want to take: '))

    start = time.time()
    t_board = [[]] * 6
    board[0] = [1,1,1,1,1,1]
    print(board)
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
'''
    label = create_uci_labels()
    create_ucilabels()
