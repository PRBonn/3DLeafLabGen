import random
import numpy as np 
from modules.leaf import Leaf
from utils.probabilities import probability_stage, probability_leaves, probability_cotyledons

# macros should not be defined here but taken from some biology/physyiology plant expert thing
# we should give "type of plant" and "stage" and that will tell us shit

# MACROS
STAGE = {'early': 0,
         'small': 1,
         'medium': 2,
         'big': 3,
         'gigantic': 4
        }

AVERAGE_WIDTH = 0.11 #[m]
AVERAGE_LENGTH = 0.14 #[m]

class Plant():
    def __init__(self, growth_stage: str=None):
        # save parameters 
        if growth_stage is not None:
            self.stage = STAGE[growth_stage.lower()]
        else:
            self.stage = probability_stage()       
        
        # generate values for leaf
        self.generate()

    def generate(self):
        self.generate_num_leaves()
        self.generate_cotyledons()
        self.generate_leaf_dict()
        self.generate_leaves()

    def generate_num_leaves(self):
        self.num_leaves = probability_leaves(self.stage)
        self.check_leaves()

    def check_leaves(self):
        if self.num_leaves < 0: self.num_leaves = 0
        if self.num_leaves == 1: self.num_leaves = 2
        if self.num_leaves == 3:
            self.num_leaves += random.choice([-1,1])
    
    def generate_cotyledons(self):
        self.cotyledons = probability_cotyledons(self.num_leaves)

    def generate_leaf_dict(self):
        self.leaf_dict = {} 
        # we should be able to insert some plant specific knowledge
        self.leaf_dict['w'] = AVERAGE_WIDTH * self.stage / 2
        self.leaf_dict['l'] = AVERAGE_LENGTH * self.stage / 2

    def generate_leaves(self):
        self.leaves = {}
        for _id in range(self.num_leaves):
            self.leaves[_id] = Leaf(_id, self.leaf_dict)

    def __str__(self):
        stage = list(STAGE)[self.stage - 1]
        to_print = f'Plant of stage {stage}, it has {self.num_leaves} leaves.\n'
        for _, leaf in self.leaves.items():
            to_print += str(leaf)  
        return to_print

