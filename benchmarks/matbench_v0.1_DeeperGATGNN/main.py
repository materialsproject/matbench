import os
import argparse
import time
import csv
import sys
import json
import random
import numpy as np
import pprint
import yaml

import torch
import torch.multiprocessing as mp

import ray
from ray import tune

from matdeeplearn import models, process, training

################################################################################
#
################################################################################
#  MatDeepLearn code
################################################################################
#
################################################################################
def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    np.random.seed(1234)
    random.seed(1234)
    start_time = time.time()
    print("Starting...")
    print(
        "GPU is available:",
        torch.cuda.is_available(),
        ", Quantity: ",
        torch.cuda.device_count(),
    )

    parser = argparse.ArgumentParser(description="MatDeepLearn inputs")
    ###Job arguments
    parser.add_argument(
        "--config_path",
        default="config.yml",
        type=str,
        help="Location of config file (default: config.json)",
    )
    parser.add_argument(
        "--run_mode",
        default=None,
        type=str,
        help="run modes: Training, Predict, Repeat, CV, Hyperparameter, Ensemble, Analysis",
    )
    parser.add_argument(
        "--job_name",
        default=None,
        type=str,
        help="name of your job and output files/folders",
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="CGCNN_demo, MPNN_demo, SchNet_demo, MEGNet_demo, GCN_net_demo, SOAP_demo, SM_demo",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for data split, 0=random",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="path of the model .pth file",
    )
    parser.add_argument(
        "--save_model",
        default=None,
        type=str,
        help="Save model",
    )
    parser.add_argument(
        "--load_model",
        default=None,
        type=str,
        help="Load model",
    )
    parser.add_argument(
        "--write_output",
        default=None,
        type=str,
        help="Write outputs to csv",
    )
    parser.add_argument(
        "--parallel",
        default=None,
        type=str,
        help="Use parallel mode (ddp) if available",
    )
    parser.add_argument(
        "--reprocess",
        default=None,
        type=str,
        help="Reprocess data since last run",
    )
    ###Processing arguments
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="Location of data containing structures (json or any other valid format) and accompanying files",
    )
    parser.add_argument("--format", default=None, type=str, help="format of input data")
    ###Training arguments
    parser.add_argument("--train_ratio", default=None, type=float, help="train ratio")
    parser.add_argument(
        "--val_ratio", default=None, type=float, help="validation ratio"
    )
    parser.add_argument("--test_ratio", default=None, type=float, help="test ratio")
    parser.add_argument(
        "--verbosity", default=None, type=int, help="prints errors every x epochs"
    )
    parser.add_argument(
        "--target_index",
        default=None,
        type=int,
        help="which column to use as target property in the target file",
    )
    ###Model arguments
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="number of total epochs to run",
    )
    parser.add_argument("--batch_size", default=None, type=int, help="batch size")
    parser.add_argument("--lr", default=None, type=float, help="learning rate")
    parser.add_argument(
        "--gc_count",
        default=None,
        type=int,
        help="number of gc layers",
    )
    parser.add_argument(
        "--dropout_rate",
        default=None,
        type=float,
        help="dropout rate",
    )

    ##Get arguments from command line
    args = parser.parse_args(sys.argv[1:])

    ##Open provided config file
    assert os.path.exists(args.config_path), (
        "Config file not found in " + args.config_path
    )
    with open(args.config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    ##Update config values from command line
    if args.run_mode != None:
        config["Job"]["run_mode"] = args.run_mode
    run_mode = config["Job"].get("run_mode")
    config["Job"] = config["Job"].get(run_mode)
    if config["Job"] == None:
        print("Invalid run mode")
        sys.exit()

    if args.job_name != None:
        config["Job"]["job_name"] = args.job_name
    if args.model != None:
        config["Job"]["model"] = args.model
    if args.seed != None:
        config["Job"]["seed"] = args.seed
    if args.model_path != None:
        config["Job"]["model_path"] = args.model_path
    if args.load_model != None:
        config["Job"]["load_model"] = args.load_model
    if args.save_model != None:
        config["Job"]["save_model"] = args.save_model
    if args.write_output != None:
        config["Job"]["write_output"] = args.write_output
    if args.parallel != None:
        config["Job"]["parallel"] = args.parallel
    if args.reprocess != None:
        config["Job"]["reprocess"] = args.reprocess

    if args.data_path != None:
        config["Processing"]["data_path"] = args.data_path
    if args.format != None:
        config["Processing"]["data_format"] = args.format

    if args.train_ratio != None:
        config["Training"]["train_ratio"] = args.train_ratio
    if args.val_ratio != None:
        config["Training"]["val_ratio"] = args.val_ratio
    if args.test_ratio != None:
        config["Training"]["test_ratio"] = args.test_ratio
    if args.verbosity != None:
        config["Training"]["verbosity"] = args.verbosity
    if args.target_index != None:
        config["Training"]["target_index"] = args.target_index

    for key in config["Models"]:
        if args.epochs != None:
            config["Models"][key]["epochs"] = args.epochs
        if args.batch_size != None:
            config["Models"][key]["batch_size"] = args.batch_size
        if args.lr != None:
            config["Models"][key]["lr"] = args.lr
        if args.gc_count != None:
            config["Models"][key]["gc_count"] = args.gc_count
        if args.dropout_rate != None:
            config["Models"][key]["dropout_rate"] = args.dropout_rate

    if run_mode == "Predict":
        config["Models"] = {}
    elif run_mode == "Ensemble":
        config["Job"]["ensemble_list"] = config["Job"]["ensemble_list"].split(",")
        models_temp = config["Models"]
        config["Models"] = {}
        for i in range(0, len(config["Job"]["ensemble_list"])):
            config["Models"][config["Job"]["ensemble_list"][i]] = models_temp.get(
                config["Job"]["ensemble_list"][i]
            )
    else:
        config["Models"] = config["Models"].get(config["Job"]["model"])

    if config["Job"]["seed"] == 0:
        config["Job"]["seed"] = np.random.randint(1, 1e6)

    ##Print and write settings for job
    print("Settings: ")
    pprint.pprint(config)
    with open(str(config["Job"]["job_name"]) + "_settings.txt", "w") as log_file:
        pprint.pprint(config, log_file)

    ################################################################################
    #  Begin processing
    ################################################################################

    if run_mode != "Hyperparameter":

        process_start_time = time.time()

        dataset = process.get_dataset(
            config["Processing"]["data_path"],
            config["Training"]["target_index"],
            config["Job"]["reprocess"],
            config["Processing"],
        )

        print("Dataset used:", dataset)
        print(dataset[0])

        print("--- %s seconds for processing ---" % (time.time() - process_start_time))

    ################################################################################
    #  Training begins
    ################################################################################
    
    ##Regular training
    if run_mode == "Training":

        print("Starting regular training")
        print(
            "running for "
            + str(config["Models"]["epochs"])
            + " epochs"
            + " on "
            + str(config["Job"]["model"])
            + " model"
        )
        world_size = torch.cuda.device_count()
        if world_size == 0:
            print("Running on CPU - this will be slow")
            training.train_regular(
                "cpu",
                world_size,
                config["Processing"]["data_path"],
                config["Job"],
                config["Training"],
                config["Models"],
            )

        elif world_size > 0:
            if config["Job"]["parallel"] == "True":
                print("Running on", world_size, "GPUs")
                mp.spawn(
                    training.train_regular,
                    args=(
                        world_size,
                        config["Processing"]["data_path"],
                        config["Job"],
                        config["Training"],
                        config["Models"],
                    ),
                    nprocs=world_size,
                    join=True,
                )
            if config["Job"]["parallel"] == "False":
                print("Running on one GPU")
                training.train_regular(
                    "cuda",
                    world_size,
                    config["Processing"]["data_path"],
                    config["Job"],
                    config["Training"],
                    config["Models"],
                )

    ##Predicting from a trained model
    elif run_mode == "Predict":

        print("Starting prediction from trained model")
        train_error = training.predict(
            dataset, config["Training"]["loss"], config["Job"]
        )
        print("Test Error: {:.5f}".format(train_error))

    ##Running n fold cross validation
    elif run_mode == "CV":

        print("Starting cross validation")
        print(
            "running for "
            + str(config["Models"]["epochs"])
            + " epochs"
            + " on "
            + str(config["Job"]["model"])
            + " model"
        )
        world_size = torch.cuda.device_count()
        if world_size == 0:
            print("Running on CPU - this will be slow")
            training.train_CV(
                "cpu",
                world_size,
                config["Processing"]["data_path"],
                config["Job"],
                config["Training"],
                config["Models"],
            )

        elif world_size > 0:
            if config["Job"]["parallel"] == "True":
                print("Running on", world_size, "GPUs")
                mp.spawn(
                    training.train_CV,
                    args=(
                        world_size,
                        config["Processing"]["data_path"],
                        config["Job"],
                        config["Training"],
                        config["Models"],
                    ),
                    nprocs=world_size,
                    join=True,
                )
            if config["Job"]["parallel"] == "False":
                print("Running on one GPU")
                training.train_CV(
                    "cuda",
                    world_size,
                    config["Processing"]["data_path"],
                    config["Job"],
                    config["Training"],
                    config["Models"],
                )

    ##Running repeated trials
    elif run_mode == "Repeat":
        print("Repeat training for " + str(config["Job"]["repeat_trials"]) + " trials")
        training.train_repeat(
            config["Processing"]["data_path"],
            config["Job"],
            config["Training"],
            config["Models"],
        )

    ##Hyperparameter optimization
    elif run_mode == "Hyperparameter":

        print("Starting hyperparameter optimization")
        print(
            "running for "
            + str(config["Models"]["epochs"])
            + " epochs"
            + " on "
            + str(config["Job"]["model"])
            + " model"
        )

        ##Reprocess here if not reprocessing between trials
        if config["Job"]["reprocess"] == "False":
            process_start_time = time.time()

            dataset = process.get_dataset(
                config["Processing"]["data_path"],
                config["Training"]["target_index"],
                config["Job"]["reprocess"],
                config["Processing"],
            )

            print("Dataset used:", dataset)
            print(dataset[0])

            if config["Training"]["target_index"] == -1:
                config["Models"]["output_dim"] = len(dataset[0].y[0])
            # print(len(dataset[0].y))

            print(
                "--- %s seconds for processing ---" % (time.time() - process_start_time)
            )

        ##Set up search space for each model; these can subject to change
        hyper_args = {}
        dim1 = [x * 10 for x in range(1, 20)]
        dim2 = [x * 10 for x in range(1, 20)]
        dim3 = [x * 10 for x in range(1, 20)]
        batch = [x * 10 for x in range(1, 20)]
        hyper_args["SchNet_demo"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "dim3": tune.choice(dim3),
            "gnn_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(
                ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
            ),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
            "cutoff": config["Processing"]["graph_max_radius"],
        }
        hyper_args["CGCNN_demo"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "gnn_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(
                ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
            ),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
        }
        hyper_args["DEEP_GATGNN_demo"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "gnn_count": tune.choice([10, 15, 20, 25, 30, 35, 40, 45, 50]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(
                ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
            ),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
        }
        hyper_args["MPNN_demo"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "dim3": tune.choice(dim3),
            "gnn_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(
                ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
            ),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
        }
        hyper_args["MEGNet_demo"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "dim3": tune.choice(dim3),
            "gnn_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
        }
        hyper_args["GCN_demo"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "gnn_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(
                ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
            ),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
        }
        hyper_args["SOAP_demo"] = {
            "dim1": tune.choice(dim1),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
            "nmax": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "lmax": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "sigma": tune.uniform(0.1, 2.0),
            "rcut": tune.uniform(1.0, 10.0),
        }
        hyper_args["SM_demo"] = {
            "dim1": tune.choice(dim1),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "lr": tune.loguniform(1e-4, 0.05),
            "batch_size": tune.choice(batch),
        }

        ##Run tune setup and trials
        best_trial = training.tune_setup(
            hyper_args[config["Job"]["model"]],
            config["Job"],
            config["Processing"],
            config["Training"],
            config["Models"],
        )

        ##Write hyperparameters to file
        hyperparameters = best_trial.config["hyper_args"]
        hyperparameters = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in hyperparameters.items()
        }
        with open(
            config["Job"]["job_name"] + "_optimized_hyperparameters.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(hyperparameters, f, ensure_ascii=False, indent=4)

        ##Print best hyperparameters
        print("Best trial hyper_args: {}".format(hyperparameters))
        print(
            "Best trial final validation error: {:.5f}".format(
                best_trial.last_result["loss"]
            )
        )

    ##Ensemble mode using simple averages
    elif run_mode == "Ensemble":

        print("Starting simple (average) ensemble training")
        print("Ensemble list: ", config["Job"]["ensemble_list"])
        training.train_ensemble(
            config["Processing"]["data_path"],
            config["Job"],
            config["Training"],
            config["Models"],
        )

    ##Analysis mode
    ##NOTE: this only works for "early" pooling option, because it assumes the graph-level features are plotted, not the node-level ones
    elif run_mode == "Analysis":
        print("Starting analysis of graph features")

        ##dict for the tsne settings; please refer to sklearn.manifold.TSNE for information on the function arguments
        tsne_args = {
            "perplexity": 50,
            "early_exaggeration": 12,
            "learning_rate": 300,
            "n_iter": 5000,
            "verbose": 1,
            "random_state": 42,
        }
        ##this saves the tsne output as a csv file with: structure id, y, tsne 1, tsne 2 as the columns
        ##Currently only works if there is one y column in targets.csv
        training.analysis(
            dataset,
            config["Job"]["model_path"],
            tsne_args,
        )

    else:
        print("No valid mode selected, try again")

    print("--- %s total seconds elapsed ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
