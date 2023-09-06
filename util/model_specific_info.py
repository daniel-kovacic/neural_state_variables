class ModelSpecificInfo:
    def __init__(self, num_of_frames, latent_enc_shape, dim):
        """
        creates ModelSpecificInfo object holding all information necessary for a model

        Parameters
        ----------
        num_of_frames: int
            number of frames used as input for dynamics prediction autoencoder
        latent_enc_shape: tuple-int
            shape of the latent encoding of a given dynamics prediction autoencoder
        dim: int
            defines in which way the frames are appended 2 (x-axis) or 3(time-axis)
        """

        self.num_of_frames = num_of_frames
        self.latent_enc_shape = latent_enc_shape
        self.dim = dim
        self.neural_state_dim = None
