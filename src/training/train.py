from src.training.data_augmentation import generate_data_augmentation


def train_model(model, model_type, dataset, epochs, batch_size, steps_per_epoch, n_batches_prof, n_batches_augmented,
                data_augmentation=True, desync=False, gaussian_noise=False, time_warping=False):
    if data_augmentation:
        da_method = generate_data_augmentation(dataset.x_profiling, dataset.y_profiling, batch_size, model_type, n_batches_prof,
                                               n_batches_augmented, desync=desync, gaussian_noise=gaussian_noise, time_warping=time_warping)
        history = model.fit_generator(
            generator=da_method,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=2,
            validation_data=(dataset.x_attack, dataset.y_attack),
            validation_steps=1,
            callbacks=[])
    else:
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
