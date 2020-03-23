class Config(object):
    def __init__(self):
        # hyperparameter
        self.struct = [4973, 4000, 3500]
        self.reg = 10
        self.alpha = 1

        # parameters for training
        self.b_s = 128
        #self.num_sampled = 10
        self.max_iters = 210
        self.sg_learning_rate = 1e-4
        self.ae_learning_rate = 1e-4
