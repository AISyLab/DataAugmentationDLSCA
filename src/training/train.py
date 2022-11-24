from src.training.data_augmentation import generate_data_augmentation


def train_model(model, dataset, epochs, batch_size):
    history = model.fit(
        x=dataset.x_profiling,
        y=dataset.y_profiling,
        batch_size=batch_size,
        verbose=2,
        epochs=epochs,
        shuffle=True,
        validation_data=(dataset.x_attack, dataset.y_attack),
        callbacks=[])

    return model, history


def train_model_augmentation(model, model_type, dataset, epochs, batch_size, steps_per_epoch, n_batches_prof, n_batches_augmented,
                             desync_level_augmentation, std_augmentation, data_augmentation_per_epoch=True,
                             augmented_traces_only=False, desync=False, gaussian_noise=False):
    da_method = generate_data_augmentation(dataset.x_profiling, dataset.y_profiling, batch_size, model_type, n_batches_prof,
                                           n_batches_augmented, desync_level_augmentation, std_augmentation,
                                           data_augmentation_per_epoch=data_augmentation_per_epoch,
                                           augmented_traces_only=augmented_traces_only,
                                           desync=desync, gaussian_noise=gaussian_noise)
    history = model.fit_generator(
        generator=da_method,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=2,
        validation_data=(dataset.x_attack, dataset.y_attack),
        validation_steps=int(len(dataset.x_attack) / batch_size),
        callbacks=[])

    return model, history
