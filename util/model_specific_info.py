class ModelSpecificInfo:
    def __init__(self, num_of_frames, latent_enc_shape, dim):
        self.num_of_frames = num_of_frames
        self.latent_enc_shape = latent_enc_shape
        self.dim = dim
        self.neural_state_dim = None

