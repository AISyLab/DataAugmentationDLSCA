def get_dataset_filepath(trace_folder, dataset_name, npoi):
    dataset_dict = {
        "ascad-variable": {
            1400: f"{trace_folder}/ascad-variable.h5",
            2000: f"{trace_folder}/ASCADr/ascad-variable_nopoi/ascad-variable_nopoi_window_20_2000_npoi.h5",
        },
        "dpa_v42": {
            2000: f"{trace_folder}/dpa_v42_nopoi_window_10_trim_2000_npoi.h5",
        },
        "ascadv2": {
            1000: f"{trace_folder}/ascadv2-extracted.h5",
        }
    }

    return dataset_dict[dataset_name][npoi]
