

class MemorySuper(object):
    def __init__(self, maxlen):
        self.memory_type = None
        pass

    def append(self, experience):
        pass

    def sample(self, batch_size):
        pass

    def batch_update(self, tree_idx, abs_errors):
        pass

    def len(self):
        pass
