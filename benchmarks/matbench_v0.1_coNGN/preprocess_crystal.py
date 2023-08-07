from run import MatbenchDataset
from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell, VoronoiAsymmetricUnitCell
import sys
from pathlib import Path

if __name__ == '__main__':
    if len(sys.argv) > 1:
        matbench_datasets_subset = sys.argv[1:]
    else:
        matbench_datasets_subset = [
            "matbench_mp_e_form", "matbench_mp_gap", 
            "matbench_mp_is_metal", "matbench_perovskites",
            "matbench_log_kvrh", "matbench_log_gvrh", "matbench_dielectric", "matbench_phonons",
            "matbench_jdft2d"]

    dataset_cache = Path('./dataset_cache/')
    mb_dataset_cache = MatbenchDataset(dataset_cache)

    # crystal_preprocessor = KNNAsymmetricUnitCell(24)
    crystal_preprocessor = VoronoiAsymmetricUnitCell(1e-6)
    for task in matbench_datasets_subset:
        preprocessed_crystals_file = mb_dataset_cache.get_dataset_file(task, crystal_preprocessor, processes=16,
                                                                       batch_size=2000)