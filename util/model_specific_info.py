class ModelSpecificInfo:
    def __int__(self, number_of_frames=2, latent_dim_shape=(2, 1, 1, 64), dim=3, latent_filename="points"):
        self.number_of_frames = number_of_frames
        self.latent_dim_shape = latent_dim_shape
        self.dim = dim
        self.latent_filename = latent_filename
