#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
import re, sys, time
from itertools import count
from collections import namedtuple
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from csv import writer
import csv
from os import walk
import os

###############################################################################
# Piece-Square tables. Tune these to change sunfish's behaviour
###############################################################################

piece = { 'P': 100, 'N': 280, 'B': 320, 'R': 479, 'Q': 929, 'K': 60000 }
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x+piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i*8:i*8+8]) for i in range(8)), ())
    pst[k] = (0,)*20 + pst[k] + (0,)*20

###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' rnbqkbnr\n'  #  20 - 29
    ' pppppppp\n'  #  30 - 39
    ' ........\n'  #  40 - 49
    ' ........\n'  #  50 - 59
    ' ........\n'  #  60 - 69
    ' ........\n'  #  70 - 79
    ' PPPPPPPP\n'  #  80 - 89
    ' RNBQKBNR\n'  #  90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)

# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    'P': (N, N+N, N+W, N+E),
    'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    'B': (N+E, S+E, S+W, N+W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
    'K': (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
MATE_LOWER = piece['K'] - 10*piece['Q']
MATE_UPPER = piece['K'] + 10*piece['Q']

# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 0

# Constants for tuning search
QS_LIMIT = 219
EVAL_ROUGHNESS = 13
DRAW_TEST = False

###############################################################################
# Chess logic
###############################################################################

class Position(namedtuple('Position', 'board score wc bc ep kp')):
    """ A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    """

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        for i, p in enumerate(self.board):
            if not p.isupper(): continue
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper(): break
                    # Pawn move, double move and capture
                    if p == 'P' and d in (N, N+N) and q != '.': break
                    if p == 'P' and d == N+N and (i < A1+N or self.board[i+N] != '.'): break
                    if p == 'P' and d in (N+W, N+E) and q == '.' \
                            and j not in (self.ep, self.kp, self.kp-1, self.kp+1): break
                    # Move it
                    yield (i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in 'PNK' or q.islower(): break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j+E] == 'K' and self.wc[0]: yield (j+E, j+W)
                    if i == H1 and self.board[j+W] == 'K' and self.wc[1]: yield (j+W, j+E)

    def rotate(self):
        ''' Rotates the board, preserving enpassant '''
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119-self.ep if self.ep else 0,
            119-self.kp if self.kp else 0)

    def nullmove(self):
        ''' Like rotate, but clears ep and kp '''
        return Position(
            self.board[::-1].swapcase(), -self.score,
            self.bc, self.wc, 0, 0)

    def move(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i+1:]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, '.')
        # Castling rights, we move the rook or capture the opponent's
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling
        if p == 'K':
            wc = (False, False)
            if abs(j-i) == 2:
                kp = (i+j)//2
                board = put(board, A1 if j < i else H1, '.')
                board = put(board, kp, 'R')
        # Pawn promotion, double move and en passant capture
        if p == 'P':
            if A8 <= j <= H8:
                board = put(board, j, 'Q')
            if j - i == 2*N:
                ep = i + N
            if j == self.ep:
                board = put(board, j+S, '.')
        # We rotate the returned position, so it's ready for the next player
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q.islower():
            score += pst[q.upper()][119-j]
        # Castling check detection
        if abs(j-self.kp) < 2:
            score += pst['K'][119-j]
        # Castling
        if p == 'K' and abs(i-j) == 2:
            score += pst['R'][(i+j)//2]
            score -= pst['R'][A1 if j < i else H1]
        # Special pawn stuff
        if p == 'P':
            if A8 <= j <= H8:
                score += pst['Q'][j] - pst['P'][j]
            if j == self.ep:
                score += pst['P'][119-(j+S)]
        return score

###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple('Entry', 'lower upper')

class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.leafNodes = 0
        self.totalNodes = 0

    def getLeafNodeCount(self):
        leafNodeCount = self.leafNodes
        NodeCount = self.totalNodes
        self.leafNodes = 0
        self.NodeCount = 0

        return leafNodeCount,NodeCount

    def bound(self, pos, gamma, depth, root=True):
        """ returns r  
                s(pos) <= r < gamma    if gamma > s(pos)
                gamma <= r <= s(pos)   if gamma <= s(pos)"""
        self.totalNodes += 1
        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the              transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        if pos.score <= -MATE_LOWER:
            self.leafNodes += 1
            return -MATE_UPPER

        # We detect 3-fold captures by comparing against previously
        # _actually played_ positions.
        # Note that we need to do this before we look in the table, as the
        # position may have been previously reached with a different score.
        # This is what prevents a search instability.
        # FIXME: This is not true, since other positions will be affected by
        # the new values for all the drawn positions.

        ##this  never called since i set to zero
        if DRAW_TEST:
            if not root and pos in self.history:
                return 0

        # Look in the table if we have already searched this position before.
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        entry = self.tp_score.get((pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        # print(entry.lower,entry.upper)
   
        if entry.lower >= gamma and (not root or self.tp_move.get(pos) is not None):
            return entry.lower
        if entry.upper < gamma:
            return entry.upper

        # Here extensions may be added
        # Such as 'if in_check: depth += 1'

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # First try not moving at all. We only do this if there is at least one major
            # piece left on the board, since otherwise zugzwangs are too dangerous.

            if depth > 0 and not root and any(c in pos.board for c in 'RBNQ'):

                yield None, -self.bound(pos.nullmove(), 1-gamma, depth-3, root=False)

            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else.
            if depth == 0:
                self.leafNodes += 1
                yield None, pos.score
            # Then killer move. We search it twice, but the tp will fix things for us.
            # Note, we don't have to check for legality, since we've already done it
            # before. Also note that in QS the killer must be a capture, otherwise we
            # will be non deterministic.
            killer = self.tp_move.get(pos)
            if killer and (depth > 0 or pos.value(killer) >= QS_LIMIT):
                yield killer, -self.bound(pos.move(killer), 1-gamma, depth-1, root=False)
            # Then all the other moves
            for move in sorted(pos.gen_moves(), key=pos.value, reverse=True):
            #for val, move in sorted(((pos.value(move), move) for move in pos.gen_moves()), reverse=True):
                # If depth == 0 we only try moves with high intrinsic score (captures and
                # promotions). Otherwise we do all moves.
                if depth > 0 or pos.value(move) >= QS_LIMIT:
                    yield move, -self.bound(pos.move(move), 1-gamma, depth-1, root=False)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # Clear before setting, so we always have a value
                if len(self.tp_move) > TABLE_SIZE: self.tp_move.clear()
                # Save the move for pv construction and killer heuristic
                self.tp_move[pos] = move
                break

        # Stalemate checking is a bit tricky: Say we failed low, because
        # we can't (legally) move and so the (real) score is -infty.
        # At the next depth we are allowed to just return r, -infty <= r < gamma,
        # which is normally fine.
        # However, what if gamma = -10 and we don't have any legal moves?
        # Then the score is actaully a draw and we should fail high!
        # Thus, if best < gamma and best < 0 we need to double check what we are doing.
        # This doesn't prevent sunfish from making a move that results in stalemate,
        # but only if depth == 1, so that's probably fair enough.
        # (Btw, at depth 1 we can also mate without realizing.)
        if best < gamma and best < 0 and depth > 0:
            is_dead = lambda pos: any(pos.value(m) >= MATE_LOWER for m in pos.gen_moves())
            if all(is_dead(pos.move(m)) for m in pos.gen_moves()):
                in_check = is_dead(pos.nullmove())
                best = -MATE_UPPER if in_check else 0

        # Clear before setting, so we always have a value
        if len(self.tp_score) > TABLE_SIZE: self.tp_score.clear()
        # Table part 2
        if best >= gamma:
            self.tp_score[pos, depth, root] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos, depth, root] = Entry(entry.lower, best)
        return best

    def search(self, pos, history,depths):

        """ Iterative deepening MTD-bi search """
        if DRAW_TEST:
            self.history = set(history)
            # print('# Clearing table due to new history')
            self.tp_score.clear()

        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply.



        # for depth in range(depths, depths+1): (this is now unessarey as the iteration deeping is done in maingame)
        depth = depths
        # The inner loop is a binary search on the score of the position.
        # Inv: lower <= score <= upper
        # 'while lower != upper' would work, but play tests show a margin of 20 plays
        # better.
        lower, upper = -MATE_UPPER, MATE_UPPER
        while lower < upper - EVAL_ROUGHNESS:
            gamma = (lower+upper+1)//2
            score = self.bound(pos, gamma, depth)
            if score >= gamma:
                lower = score
            if score < gamma:
                upper = score
        # We want to make sure the move to play hasn't been kicked out of the table,
        # So we make another call that must always fail high and thus produce a move.
        self.bound(pos, lower, depth)
        # If the game hasn't finished we can retrieve our move from the
        # transposition table.

        yield depth, self.tp_move.get(pos), self.tp_score.get((pos, depth, True)).lower

class BotFactory():

    def createBot(self, method):
        if method == "mm":
            return mmBot(Bot)

class randomBot():
    def __init__(self):
        self.leafNodes = 0
        self.name=  "randomBot"

    def move(self, pos, depth):
        return self.randomPlayer(pos)

    def randomPlayer(self,pos):
        moves=list(pos.gen_moves())
        r=random.randint(0, len(moves)-1)
        return -1,moves[r]
    def getLeafNodeCount(self):
        return 0

class mmBot():

    def __init__(self):
        self.leafNodes = 0
        self.totalNodes = 0
        self.name = "mmBot"


    def move(self,pos,depth):
        return  self.mm(pos, depth)


    def mm(self, pos,depth):

        #here
        self.totalNodes += 1


        if depth == 0:
            self.leafNodes += 1
            return pos.score, None

        if pos.score <= -MATE_LOWER:
            self.leafNodes += 1
            return -456787654, None

        bestMove = None
        minEval = 10090909
        for move in pos.gen_moves():
            eval, m = self.mm(pos.move(move), depth - 1)
            if eval < minEval:
                minEval = eval
                bestMove = move
            # maxEval = max(eval, maxEval)
        return -minEval, bestMove

    def getLeafNodeCount(self):
        LeafNodeCount = self.leafNodes
        self.leafNodes  = 0
        return LeafNodeCount

class mmRandoBot():

    def __init__(self):
        self.leafNodes = 0
        self.name = "mmRandoBot"

    def move(self,pos,depth):
        return  self.mmrando(pos, depth,-234564322,123526225)

    def mmrando(self,pos, depth,a,b):
        if depth == 0:
            self.leafNodes+=1
            #color*color = 1
            return pos.score,None
        if pos.score <= -MATE_LOWER:
            self.leafNodes+=1
            return -7898788,None


        bestMove = None
        maxeval = -1009099
        for move in pos.gen_moves():
            #my utility is the opposite of my oppnrnts
            evalOpp, m = self.mmrando(pos.move(move), depth - 1,-b,-a )
            #so we lose quicker
            evalMy = -evalOpp- random.randint(0,0)
            #found a better move
            if evalMy  > maxeval:
                maxeval = evalMy
                bestMove = move
            # maxEval = max(eval, maxEval)
            a = max(a, evalMy )
            if a>=b:
                break

        if bestMove == None:
            bestMove =  move

        return maxeval, bestMove


    def getLeafNodeCount(self):
        LeafNodeCount = self.leafNodes
        self.leafNodes  = 0
        return LeafNodeCount

class mmabBot():
    def __init__(self):
        self.leafNodes = 0
        self.totalNodes = 0
        self.name = "mmabBot"


    def move(self, pos,depth):
        return self.mmab(pos, depth,-10000001,10000002)


    def mmab(self,pos, depth,a,b):
        self.totalNodes += 1
        if depth == 0:
            self.leafNodes+=1
            return pos.score,None
        if pos.score <= -MATE_LOWER:
            self.leafNodes+=1
            return -7898788,None
        bestMove = None
        maxeval = -MATE_LOWER
        ## This orders the moves generated, by a huristic pos.value so that more cutoffs are achived.
        # eg. if is pawn capture, try that first.
        for move in sorted(pos.gen_moves(), key=pos.value, reverse=True):
            #my utility is the opposite of my oppnrnts
            evalOpp, m = self.mmab(pos.move(move), depth - 1,-b,-a )
            evalMy = -evalOpp
            #found a better move
            if evalMy  > maxeval:
                maxeval = evalMy
                bestMove = move
            # maxEval = max(eval, maxEval)
            a = max(a, evalMy )
            if a>=b:
                break
        return maxeval, bestMove

    def getLeafNodeCount(self):
        leafNodeCount = self.leafNodes
        totalNodes = self.totalNodes
        self.leafNodes  = 0
        self.totalNodes  = 0
        return leafNodeCount, totalNodes

class negaScoutBot():
    def __init__(self):
        self.leafNodes = 0
        self.totalNodes = 0
        self.name = "negaScoutBot"

    def move(self, pos,depth):
        return self.negaScout(pos, depth,-10000001,10000002)



    def negaScout(self, pos, depth,a,b, root =True ):
        self.totalNodes += 1
        if depth == 0:
            self.leafNodes+=1
            return pos.score
        ## -mate lower dosent work here as for some reson. see repot implementation
        if pos.score <=-50710:
            self.leafNodes+=1
            return pos.score
        score= -MATE_LOWER
        n = b
        moves = []
        ## This orders the moves generated, by a huristic pos.value so that more cutoffs are achived.
        # eg. if is pawn capture, try thata first.
        for m in sorted(pos.gen_moves(), key=pos.value, reverse=True):
            moves.append(m)
        for w , move in enumerate(moves):
            cur = -self.negaScout(pos.move(move),depth - 1, -n,-a,False)
            if cur > score:
                if n == b or depth <=2:
                    score = cur
                else:
                    score = -self.negaScout(pos.move(move),depth - 1, -b,-cur,False)
            if score > a:
                a =score
                bestMove = move
            if a>b:
                return a
            n = a+1
        if root: return score, bestMove
        return score

    def getLeafNodeCount(self):
        leafNodeCount = self.leafNodes
        nodeCount = self.totalNodes

        self.leafNodes  = 0
        self.totalNodes  = 0
        return leafNodeCount, nodeCount

class mtd_biBot(Searcher):

    def __init__(self):
        self.leafNodes = 0
        self.name = "mtd_biBot"

    def move(self, pos, depth):
        searcher = Searcher()

        for _depth, move, score in searcher.search(pos, {}, depth):
            # print("mtd",_depth, move, score)
            pass


        print( "pppp",searcher.bound(pos,0, depth))
        # _depth, move, score = searcher.search(pos,{})

        self.leafNodes =searcher.getLeafNodeCount()
        return score,move

    def getLeafNodeCount(self):
        LeafNodeCount = self.leafNodes
        self.leafNodes = 0
        return LeafNodeCount


def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank

def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)

def print_pos(pos):
    print()
    uni_pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
                  'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙', '.':'⛝'}
    for i, row in enumerate(pos.board.split()):
        print(' ', 8-i, ' '.join(uni_pieces.get(p, p) for p in row))
    print('    a b c d e f g h \n\n')


###############################################################################
# User interface
###############################################################################
def showGraph(botList):
    for bot in botList:
        # Load the cvs as a df
        df = pd.read_csv(bot.name + ".csv")

        with pd.option_context('display.max_rows', None, 'display.max_columns',None):  # more options can be specified also
            print(df)

        # Get each bots mean nodes expanded at each depth
        depthLiWhite = []
        for l in df.mean(axis=0, skipna=True):
            depthLiWhite.append(l)
        # plt.plot(range(1, maxDepth + 1), depthLiWhite)

    plt.show()

def playBots(botList,maxDepth, numGamesForPlayer):
    #this mothos played each bot in bot lists vs a random player,
    #and appends the number of nodes expanded by a bot at a depth
    # is stored in a csv file named bot.name

    #
    for bot in botList:
        print("Playing ", bot.name,"vs random ","mmRandoBot.name")
        #play j games


        ##add the bot nodes expanded at each depth
        li = []
        for j in range(numGamesForPlayer):
            numNodesEpanded, winner = main_game(bot, mmRandoBot(), maxDepth)
            print("here",numNodesEpanded)
            li.append(numNodesEpanded)

        # Add bots depths to its csv file
        whiteMoveDepths = prosses(li)
        with open(bot.name+'.csv', 'a', newline='') as f_object:
            writer_object = writer(f_object)
            for depthRow in whiteMoveDepths:
                writer_object.writerow(depthRow)
            f_object.close()

        # Load the cvs as a df
        df = pd.read_csv(bot.name+".csv")

        with pd.option_context('display.max_rows', None, 'display.max_columns',None):  # more options can be specified also
            print(df)

        # Get eacch bots mean nodes expanded at each depth
        depthLiWhite = []
        for l in df.mean(axis=0, skipna=True):
            depthLiWhite.append(l)


        plt.plot(range(1,maxDepth+1), depthLiWhite)
    plt.show()

def main():
    botList= [mmBot,mmabBot(),negaScoutBot(),mtd_biBot()]
    playBots(botList,  maxDepth=6, numGamesForPlayer =3)
    showGraph(botList)

def prosses(li):
    moveDepths = []
    for game in li:
        for move in game:
            moveDepths.append(move[0])
    return moveDepths
# def mtd_bi(searcher,hist):
#     start = time.time()
#     for _depth, move, score in searcher.search(hist[-1], hist):
#         print(_depth, score, move)
#         if score == 69290:
#             print("mate found for black by theri bot ")
#             break
#         if time.time() - start > 0.000001:
#             break
#
#     return score, move
#
# def miniMax_(searcher,hist):
#     for i in range(1, 3):
#         # move, score = searcher.miniMaxCall(hist[-1], i, True)
#
#         score, move = searcher.miniMax(hist[-1], i, True, False)
#
#         print("black my bot at depth", i, move, "score:", score)
#         if score >= 69290:
#             print("my bot found  a mate")
#             break
#     return  score, move
#
#
# ###minmax
# def mm_(searcher, hist):
#     for i in range(1,4):
#         score, move =  searcher.mm(hist[-1], i)
#         print("mm black, deptyh score move", i ,score,move)
#         nodesAtDepth[i]
#
#         if score >= MATE_LOWER:
#             return score, move
#
#     return score,move
# ##minimax with alphabeta
# def mmab_(searcher,hist):
#
#     li = [0]*8
#     for i in range(1,4):
#         s,m=searcher.mmab(hist[-1], i, -10000008, 10000008)
#         li[i-1]=searcher.leafNodesMM
#
#         searcher.leafNodesMM = 0
#         print("mmab depth score move",i,s,m)
#         if s <= -55000:
#             return s, m,li
#         if s > 55000:
#             return s, m,li
#
#     return s,m,li
# ##negascot
# def pvs_(searcher,hist):
#     # bestMove= None
#     #
#     # maxScore =-100000000
#     # for i in range (1,5):
#     #     for m in hist[-1].gen_moves():
#     #
#     #         s =-searcher.pvs(hist[-1].move(m), i, -100000001, 12345677)
#     #        # s =-searcher.negaScout(hist[-1].move(m), i, -10000000, 12345677)
#     #
#     #
#     #         if s > maxScore:
#     #             maxScore = s
#     #             bestMove = m
#     #         print(maxScore, m,end = "")
#     #
#     #     if s <= -50000:
#     #         return maxScore, bestMove
#     #     if s > 55000:
#     #         return maxScore,bestMove
#     li = [0]*8
#     for i in range(1,6):
#         # maxScore,bestMove = searcher.pvs(hist[-1], i, -100000001, 12345677,True)
#
#         maxScore,bestMove = searcher.negaScout(hist[-1], i, -100000001, 12345677,True)
#         li[i-1]=searcher.leafNodesNM
#         searcher.leafNodesNM = 0
#         print( maxScore,bestMove)
#     return maxScore, bestMove,li
#
# def random_player(searcher,hist):
#     return hist[-1].score,searcher.randomPlayer(hist[-1])
#
# def nm_(searcher,hist):
#
#
#     for i in range(1,5):
#
#         s,m = searcher.nm(hist[-1], i, 1, hist,-1999341124,12345654333)
#
#
#         print("score move ddepth", s,m,i)
#
#         if s <=-50000:
#             return s,m
#         if s > 55000:
#             return s, m
#     return s,m
#
def isDraw(hist):
    ##this method was written by Oliver
    c = 0
    for po in hist[::-1]:
        if po == hist[-1]: c += 1
        if c == 3:
            return True

def poltNodes(nodesEpanded):


    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dfw)




    plt.plot([1, 2, 3], depthLiWhite)
    plt.plot([1, 2, 3 ], depthLiBlack)
    plt.legend(["Dataset 1", "Dataset 2"])

    plt.show()

def main_game(whiteBot,blackBot, plys):
    hist = [Position(initial, 0, (True,True), (True,True), 0, 0)]
    searcher = Searcher()

    numNodesEpanded = []

    while True:
        print("---------------------------------------------------")
        print_pos(hist[-1])
        if isDraw(hist):
            print("draw!!!")
            return numNodesEpanded, 0
        ##white to play

        if hist[-1].score <= -MATE_LOWER:
            print("black won")
            return numNodesEpanded, -1

        # start = time.time()
        depth=0
        nodesAtDepthW = []
        # white start - time.time()<1
        while depth < plys:
            depth+=1

            score, move = whiteBot.move(hist[-1], depth)
            ##get and reset leaf nodes
            ln, tn=whiteBot.getLeafNodeCount()
            print(whiteBot.name, "Score: ", score,"move: ", move ,"depth:",depth,"nodes expanded",tn)


            nodesAtDepthW.append(ln)
            # exit if loosing !




        ##make move for white ie. the move set in the last, deeest iteration
        hist.append(hist[-1].move(move))
        print_pos(hist[-1].rotate())

        if hist[-1].score <= -MATE_LOWER:
            print("white won")
            return numNodesEpanded, 1


        ## black to play

        # score, move,numLeafExpBlack = blackStg(searcher,hist)
        # numNodesEpanded.append([numLeafExpWhite,numLeafExpBlack])

        # start = time.time()

        depth=0
        nodesAtDepthB = []
        r = random.randint(2, 3)
        while(depth<r):
            depth+=1
            score, move = blackBot.move(hist[-1], depth)
            n=blackBot.getLeafNodeCount()

            #break if lost
            if score<=-MATE_LOWER:
                break
            print(score, move ,depth,n)
            nodesAtDepthB.append(n)
        numNodesEpanded.append([nodesAtDepthW,nodesAtDepthB])
        print("blacks move, score", move, score)
        hist.append(hist[-1].move(move))





if __name__ == '__main__':
    main()

