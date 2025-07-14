import random
import math

def probability_stage():
    return random.choice((3,4,5)) 

def probability_leaves(stage):
    return math.ceil(random.gauss(stage, stage_to_scale(stage)))

def stage_to_scale(stage):
    if stage == 0: return 0.25
    if stage == 2: return 1
    return 1.25

def probability_cotyledons(num_leaves):
    if num_leaves <= 4: return True
    if num_leaves > 8: return False
    sample = random.random()
    if sample >= 0.5: return True
    return False
