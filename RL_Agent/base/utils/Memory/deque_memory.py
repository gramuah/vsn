import random
from collections import deque
import numpy as np
from RL_Agent.base.utils.Memory.memory_super import *


class Memory(MemorySuper):
    def __init__(self, maxlen=20000):
        self.memory_type = "queue"
        self.memory = deque(maxlen=maxlen)

    def append(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return None, np.array(random.sample(self.memory, batch_size) ,dtype=object), None

    def batch_update(self, tree_idx=None, abs_errors=None):
        pass

    def len(self):
        return len(self.memory)