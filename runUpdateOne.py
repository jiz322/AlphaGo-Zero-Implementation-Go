import logging

import coloredlogs

from Coach_update_one import Coach_update_one
from go.Game import Game as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    
    'arenaCompare': 50,         # 
    'maxLevel' : 6,
    'maxLeaves': 8,
    
    
    'size': 9,                  #board size
    'numIters': 1,
    'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 0,        # zero
    'updateThreshold': 0.51,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 400,          # levelBased.
    'cpuct': 1.1,
    'instinctArena': False,     #if set true reset Arena's MTCL tree each time
    'balancedGame': True,      # if balanced, black should win over 6 scores
    'firstIter': False,        # No checkpoint for self-play
    'checkpoint': './temp/',
    'load_folder_file': ('./temp','best.pth.tar'),
    'resignThreshold': -1.1,   #resign when best Q value less than threshold Q[-1, 1]
    'levelBased': True,


})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    log.info('Loading the Coach_updateBest...')
    c = Coach_update_one(g, nnet, args)

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()

