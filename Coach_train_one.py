# Train one nn weight called temp.tar

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

class Coach_train_one():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.n2net = self.nnet.__class__(self.game)  # the competitor network
        self.n3net = self.nnet.__class__(self.game)  # the competitor network
        self.n4net = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations

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
        
        log.info(f'Starting Training  ...')
        self.trainExamplesHistory = [] 
        # examples of the iteration
        self.loadExamples() # Load the best.pth.tar.examples
        # shuffle examples before training
        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)    
        self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar') 
        shuffle(trainExamples)              
        self.nnet.train(trainExamples)
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.tar')
        log.info('WRITTEN temp.tar')
                
    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    #load examples only
    def loadExamples(self):
        folder = self.args.checkpoint
        filename = os.path.join(folder, "best.pth.tar.examples")
        log.info("File with trainExamples found. Loading it...")
        with open(filename, "rb") as f:
            count = 0
            while True:
                try:
                    if count == 0:
                        self.trainExamplesHistory = Unpickler(f).load()
                        count += 1
                    else:
                        self.trainExamplesHistory += Unpickler(f).load()
                except EOFError:
                    break 
        # while len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
        #     self.trainExamplesHistory.pop(0)
        log.info('Loading done!')
