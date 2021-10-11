# This files serves only for illustrative purposes.
# The code used for the published results can be found in the following repo: https://github.com/ml-evs/modnet-matbench

from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_KEY
from modnet.preprocessing import MODData
from modnet.featurizers.presets import DeBreuck2020Featurizer
from modnet.models import EnsembleMODNetModel
from pymatgen.core import Composition

USE_GA = False # wheter to use the GA or fit_preset (dynamic grid-search) for hyper-paremeter optimization.

mb = MatbenchBenchmark(
    autoload=False, 
    subset=[
        'matbench_dielectric', 
        'matbench_jdft2d', 
        'matbench_steels', 
        'matbench_expt_gap', 
        'matbench_phonons',
        'matbench_log_gvrh',
        'matbench_log_kvrh',
        'matbench_glass', 
        'matbench_expt_is_metal',
        #'matbench_perovskites', # for the bigger tasks, USE_GA=True is recommended, as training time scales better with larger training sets
        #'matbench_mp_e_form',
        #'matbench_mp_gap',
        #'matbench_mp_is_metal',
    ],
)

for task in mb.tasks:
    task.load()
    
    if task.metadata.task_type == CLF_KEY:
        classification = True
    else:
        classification = False

    for fold in task.folds:

        # load and create training MODData (this performs featurization + feature selection)
        train_df = task.get_train_and_val_data(fold, as_type="df")
        

        targets = [
            col for col in train_df.columns if col not in ("id", "structure", "composition")
        ]

        try:
            materials = train_df["structure"] if "structure" in train_df.columns else train_df["composition"].map(Composition)
        except KeyError:
            raise RuntimeError(f"Could not find any materials data dataset for task {task!r}!")

        fast_oxid_featurizer = DeBreuck2020Featurizer(fast_oxid=True)
        train_data = MODData(
            materials=materials.tolist(),
            targets=train_df[targets].values,
            target_names=targets,
            featurizer=fast_oxid_featurizer,
        )
        train_data.featurize(n_jobs=32)
        train_data.feature_selection(
                    n=-1, use_precomputed_cross_nmi=True
                )


        # create model
        targets_hierarchy = [[[field for field in targets]]]
        weights = {field: 1 for field in targets}
        model = EnsembleMODNetModel(targets_hierarchy, weights)

        # fit model

        if USE_GA:
            # you can either use a GA for hyper-parameter optimization or...
            from modnet.hyper_opt import FitGenetic
            ga = FitGenetic(train_data)
            model = ga.run(
                size_pop=20,
                num_generations=10,
                n_jobs=16,
                early_stopping=True,
                refit=True,
            )
        else:
            # ... a list of presets (kind of dynamic grid search)
            (
                models,
                val_losses,
                best_learning_curve,
                learning_curves,
                best_presets,
            ) = model.fit_preset(
                train_data,
                classification=classification,
                nested=5,
                n_jobs=16,
            )

        # Load and featurize test dataset
        test_df = task.get_test_data(fold, include_target=False, as_type="df")

        try:
            materials = test_df["structure"] if "structure" in test_df.columns else train_df["composition"].map(Composition)
        except KeyError:
            raise RuntimeError(f"Could not find any materials data dataset for task {task!r}!")

        test_data = MODData(
            materials=materials.tolist(),
            featurizer=fast_oxid_featurizer,
        )
        test_data.featurize(n_jobs=32)
        test_data.feature_selection(
                    n=-1, use_precomputed_cross_nmi=True
                )

        # predict on test data
        predict_kwargs = {}
        if classification:
            predict_kwargs["return_prob"] = True
        if model.can_return_uncertainty:
            predict_kwargs["return_unc"] = True

        pred_results = model.predict(test_data, **predict_kwargs)

        if isinstance(pred_results, tuple):
            pred_df, stds = pred_results
        else:
            pred_df = pred_results
            stds = None

        if classification:
            predictions = pred_df.values[:, 0].astype(bool).flatten()
        else:
            predictions = pred_df.values.flatten()

        # record predictions
        task.record(fold, predictions)


        mb.to_file("results.json.gz")