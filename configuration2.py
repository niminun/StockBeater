import os

class BeaterParams(object):

    def __init__(self):

        self.do_learn = False
        self.use_model = "snapshots/2017-06-06_45000"
        self.option = "Adj Close"
        self.train_set_ratio = 0.8
        self.records_per_sample = 30
        self.fetch_data = False
        self.fetch_data_for_prediction = True
        self.first_year = 1996
        self.train_batch_size = 32
        self.test_batch_size = 256
        self.num_iters = 50001
        self.lmbda = 1e-3  # regularization coefficient
        self.learning_rate = 1e-4
        self.summaries_dir = "./conf2/"

        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)

        with open("./s&p500.txt", "r") as f:
            self.symbols = f.read().split('\n')


