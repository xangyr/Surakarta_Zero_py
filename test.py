import numpy as np
import time

class Move():
    def __init__(self, from_x, from_y, to_x, to_y):
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y


class Chess():
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y


class GameBoard(object):
    board = [[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    blackNum = 6
    whiteNum = 6

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
                    for k in range(i-1,i+2):
                        for l in range(j-1,j+2):
                            if k >= 0 and k<=5:
                                if(l >= 0 and l<=5):
                                    if i is not k or j is not l:
                                        if game_board.board[k][l] is 0:
                                            moves.append(Move(i,j,k,l))
                                            #print(i,j,k,l)

        for _ in range(len(exrool_s)):
            try:
                exrool_s.remove([0,0,0,0,0,0])
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
            exrool_chess.append(Chess(game_board.board[index][i],index,i))
        exrool_s.append(exrool)
        exrool_chess_s.append(exrool_chess)

        exrool = []
        exrool_chess = []
        for i in range(6):
            exrool.append(game_board.board[i][5-index])
            exrool_chess.append(Chess(game_board.board[i][5-index],i,5-index))
        exrool_s.append(exrool)
        exrool_chess_s.append(exrool_chess)

        exrool = []
        exrool_chess = []
        for i in reversed(range(6)):
            exrool.append(game_board.board[5-index][i])
            exrool_chess.append(Chess(game_board.board[5-index][i],5-index,i))
        exrool_s.append(exrool)
        exrool_chess_s.append(exrool_chess)

        exrool = []
        exrool_chess = []
        for i in reversed(range(6)):
            exrool.append(game_board.board[i][index])
            exrool_chess.append(Chess(game_board.board[i][index],i,index))
        exrool_s.append(exrool)
        exrool_chess_s.append(exrool_chess)

        for _ in range(len(exrool_s)):
            try:
                exrool_chess_s.pop(exrool_s.index([0,0,0,0,0,0]))
                exrool_s.remove([0,0,0,0,0,0])
            except:
                pass

        return exrool_s,exrool_chess_s

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
                if rools[(i + 1)%(len(rools))][j] is -who:
                    p_n = j
                    break
                elif rools[(i + 1)%(len(rools))][j] is who:
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
                            if [rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[(i + 1) % (len(rools))][p_n].x,
                                    rools_chess[(i + 1) % (len(rools))][p_n].y] not in moves:
                                moves.append([rools_chess[i][p].x, rools_chess[i][p].y,
                                                  rools_chess[(i + 1) % (len(rools))][p_n].x,
                                                  rools_chess[(i + 1) % (len(rools))][p_n].y])
                except:
                    pass

            else:
                if [rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[(i + 1)%(len(rools))][p_n].x,
                                  rools_chess[(i + 1)%(len(rools))][p_n].y] not in moves:
                    moves.append([rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[(i + 1)%(len(rools))][p_n].x,
                                  rools_chess[(i + 1)%(len(rools))][p_n].y])

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
                                if [rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[i - 1][p_n].x,
                                        rools_chess[i - 1][p_n].y] not in moves:
                                    moves.append(
                                        [rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[i - 1][p_n].x,
                                             rools_chess[i - 1][p_n].y])
                except:
                    pass
            else:

                if [rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[i - 1][p_n].x,
                        rools_chess[i - 1][p_n].y] not in moves:
                    moves.append([rools_chess[i][p].x, rools_chess[i][p].y, rools_chess[i - 1][p_n].x,
                                      rools_chess[i - 1][p_n].y])

        return moves
    
    
    
#gameBoard = GameBoard()
#gameBoard.print_board(gameBoard.board)
#gameBoard.move_generate(gameBoard, -1)

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

    print('Run test for move generator in',time.time()-start)