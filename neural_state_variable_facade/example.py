from facade import *

if __name__ == "__main__":
    # create dataset_info which includes most information about the dataset and used model.
    # Here 2d/3d convolution and the number of frames can be selected.
    dataset_info = create_dataset_info("double_pendulum", 60, 1100, dim=3, num_of_frames=2)

    # loads the corresponding untrained dynamics prediction autoencoder depending on the chosen value of dim
    dyn_pred_autoencoder = load_untrained_dynamics_prediction_autoencoder(dataset_info)
    dyn_pred_autoencoder.summary()

    # get the RGB encoded input and optionally expected output frames as numpy array
    # shape=(number_of_vids, frames_per_vid, 128, 128, 3)
    # It is not possible to use videos with different number of frames
    # The data in this case should be stored as at ./memmap_data/{dataset_name}
    # data = get_memmap_data(dataset_info.get_memmap_path(), dataset_info)
    # loads optional hidden data from
    # The data in this case should be stored as at ./memmap_data/{dataset_name}_hidden
    # hidden_data = get_memmap_data(dataset_info.get_memmap_path(hidden=True), dataset_info)

    dataset_generator = get_dataset_generator(dataset_info)

    # use the data and optionally the hidden data arrays to create tf.dataset objects. Alternatively .png images can be
    # used if they are saved in the required form.
    # dataset_generator = get_dataset_generator(dataset_info, data_array=data)
    # hidden_data_array=hidden_data, has_hidden_parts=True)
    train_dataset = dataset_generator(mode="train")
    val_dataset = dataset_generator(mode="val")
    full_dataset = dataset_generator(mode="all", sequential=True, one_vid_per_batch=True)



    # the dynamic prediction autoencoder is trained and the results visualized, optionally a save path can be provided
    # to save the training progress between epochs.
    # The training duration needs to be adjusted to balance accuracy and training duration
    train_dynamics_prediction_autoencoder(
        train_dataset, val_dataset, dyn_pred_autoencoder, epochs=50, steps_per_epoch=200)
    visualize_single_prediction(dyn_pred_autoencoder, val_dataset, dataset_info)

    # The latent encoding for each datapoint in the dataset is produced and returned as a numpy array
    # if the memmap argument is set to True the result is stored at ../latent_data/{dataset_name}/{filename}
    # as np.memmep
    latent_encoding = create_latent_space_predictions(dataset_info, full_dataset, dyn_pred_autoencoder)

    # The intrinsic dimension of the latent encoded data is determined using the Levina Bickel's algorithm
    # this is used to approximate the (neural) state dimension. The result is also stored inside the dataset_info object
    intrinsic_dim = find_intrinsic_dimension(dataset_info, latent_encoding)
    print(f"The intrinsic un-rounded dimension approximation found by the Levina Bickel's alggorithm is: "
          f" {intrinsic_dim}")

    # a latent rec autoencoder with a latent dimension equal to the found intrinsic dimension is created
    latent_rec_autoencoder = load_untrained_latent_reconstruction_autoencoder(dataset_info)
    latent_rec_autoencoder.summary()

    # new tf.data.Datasets are created from the latent encoding
    dataset_generator = get_dataset_generator(dataset_info, data_array=latent_encoding, latent=True)
    latent_train_dataset = dataset_generator(mode="train")
    latent_val_dataset = dataset_generator(mode="val")
    latent_full_dataset = dataset_generator(mode="all", sequential=True, one_vid_per_batch=True)

    # the latent reconstruction autoencoder is trained and the result visualized
    # optionally a save path can be added to save the training progress after epochs
    train_latent_rec_autoencoder(
        latent_train_dataset, latent_val_dataset, latent_rec_autoencoder, epochs=100, steps_per_epoch=1700)
    visualize_single_prediction(
        dyn_pred_autoencoder, val_dataset, dataset_info, latent_rec_autoencoder=latent_rec_autoencoder)

    # create longterm prediction and store them as frames and videos
    make_longterm_prediction(dataset_info, dyn_pred_autoencoder, latent_rec_autoencoder=latent_rec_autoencoder)

