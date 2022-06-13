import logging

import coloredlogs

from Coach_train_one import Coach_train_one
from go.Game import Game as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({    
    'size': 9,                  #board size
    'checkpoint': './temp/',
    'load_folder_file': ('./temp','best.pth.tar'),
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    log.info('Loading the Coach_updateBest...')
    c = Coach_train_one(g, nnet, args)

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()

