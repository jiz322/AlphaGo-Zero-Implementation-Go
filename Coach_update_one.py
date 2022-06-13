import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
from tqdm import tqdm
from Arena import Arena
from MCTS import MCTS
log = logging.getLogger(__name__)

class Coach_update_one():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.firstIter = args.firstIter #set true if it produce first chechpoint to save
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.

        Go edit:
        load best.pth.tar to compare and self-iteration
        temp.pth.tar only for training. it may be overwrite shortly
        """

        log.info(f'Starting Update Competation ...')

        self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.tar') 
        nmcts = MCTS(self.game, self.nnet, self.args)
        self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar') #Load the best after training to maximize efficenty
        pmcts = MCTS(self.game, self.pnet, self.args)
        log.info('Arena Start! A tournament between PREVIOUS VERSION and the CURRENT one!')

        player_prev = lambda x: np.argmax(pmcts.getActionProb(x, arena=1, temp=0,instinctPlay=self.args.instinctArena, levelBased=self.args.levelBased)[0])
        player_n = lambda x: np.argmax(nmcts.getActionProb(x, arena=1, temp=0,instinctPlay=self.args.instinctArena, levelBased=self.args.levelBased)[0])
        playerList = [player_prev, player_n]
        nnList = [self.pnet, self.nnet]
        result = self.tournament(playerList)
        if result[0] < result[1]: 
            winner = nnList[result.index(max(result))]
            winner.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            log.info('UPDATING BEST')
            log.info('ARENA RESULT: ', result)
        log.info('PREVIOUS WIN')
        log.info('ARENA RESULT: ', result)

                
    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def playGame(self, player1, str1, player2, str2):
        arena = Arena(player1, player2, self.game)
        x, y, z, xb = arena.playGames(self.args.arenaCompare, verbose=False)
        print(str1, " win: ", x)
        print(str2, " win: ", y)
        print(str1, " win black: ", xb)
        return x, y

    def tournament(self, playList):
        tournamentResult = dict.fromkeys(playList, 0)
        previousWin, currentWin = self.playGame(playList[0], 'prev', playList[1], 'curr')  
        # tournamentResult[playList[0]] += (previousWin - currentWin + self.args.arenaCompare)/2
        # tournamentResult[playList[1]] += (currentWin - previousWin + self.args.arenaCompare)/2
        tournamentResult[playList[0]] += previousWin
        tournamentResult[playList[1]] += currentWin
        return list(tournamentResult.values())