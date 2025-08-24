

class bpe_model_train_config:
    def __init__(self):
        self.epochs = 2
        self.batch_size = 128
        self.block_size = 128
        self.n_embd = 128
        self.n_layers = 4
        self.n_heads = 4
        self.lr = 3e-4
        self.eval_interval = 200

def get_training_configs():
    return bpe_model_train_config()