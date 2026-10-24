from chgnet.data.dataset_custom import StructureDataCustomProperty, get_train_val_test_loader, get_loader
from chgnet.trainer.trainer_custom_prop import Trainer
from chgnet.model.model_custom import CHGNetCustomProperty
from matbench.bench import MatbenchBenchmark
import argparse
import torch
import os
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get input for chgnet")
    parser.add_argument("--fold", help="fold number")
    parser.add_argument("--dataset_name", help="task name")
    parser.add_argument("--batch_size", help="batch size")
    parser.add_argument("--lr", help="learning rate")
    parser.add_argument("--epoch", help="number of epochs")
    parser.add_argument("--torch_seed", help="random seed for pytorch")
    parser.add_argument("--data_seed", help="random seed for data")
    parser.add_argument("--wandb", help="wandb path")
    parser.add_argument("--save", help="save path")
    args = parser.parse_args()

    # Load Matbench
    mb = MatbenchBenchmark(autoload=False,subset=[args.dataset_name])
    for task in mb.tasks:
        task.load()

    # Obtain dataset
    dataset_train_val = StructureDataCustomProperty.from_matbench(
        fold=int(args.fold), dataset_name=args.dataset_name, is_train=True  
    )
    dataset_test = StructureDataCustomProperty.from_matbench(
        fold=int(args.fold), dataset_name=args.dataset_name, is_train=False  
    )

    # Obtain dataloader. Use the full training data as validation set and tune hyperparameters on it.
    train_loader, _ = get_train_val_test_loader(
        dataset_train_val, batch_size=int(args.batch_size), train_ratio=1.0, val_ratio=0.0, return_test=False
    ) 
    val_loader, _ = get_train_val_test_loader(
        dataset_train_val, batch_size=int(args.batch_size), train_ratio=1.0, val_ratio=0.0, return_test=False
    ) 
    test_loader = get_loader(dataset_test, batch_size=int(args.batch_size))

    # Load pretrained model
    model = CHGNetCustomProperty.load_w_attn(attn_readout_is_average=False)

    trainer = Trainer(
        model=model,
        targets="c",
        optimize_prediction_head_only = False, #True,
        optimizer="Adam",
        scheduler="CosLR", #"CosLR", 
        criterion="MSE",
        learning_rate=float(args.lr),
        epochs=int(args.epoch),
        print_freq = 10,
        use_device="cuda",
        torch_seed = int(args.torch_seed),
        data_seed = int(args.data_seed),
        wandb_path = args.wandb,
    )

    trainer.train(train_loader, val_loader, test_loader, is_normalized=True, save_dir=args.save)

    # Load the best model
    for file in os.listdir(args.save):
        if file.startswith("bestE_"):
            checkpoints = torch.load(os.path.join(args.save, file))
            break
    trainer.model.load_state_dict(checkpoints["model"]["state_dict"])
    trainer.normalizer.load_state_dict(checkpoints["normalizer"])
    trainer.model.eval()

    # Evaluate on test set
    predictions = []
    with torch.no_grad():
        for idx, (graphs, targets) in enumerate(test_loader):
            # get input
            graphs = [g.to(trainer.device) for g in graphs]
                
            # compute output
            prediction = model(graphs, task="c") 
            prediction = {k: trainer.normalizer.denorm(v) for k, v in prediction.items()}
            out = prediction["c"].detach().cpu().numpy().reshape(-1,1)
            predictions.append(out)   
    predictions = np.concatenate(predictions, axis=0)
    task.record(int(args.fold), predictions)

    mb.to_file(os.path.join(args.save, f"chgnet_finetune_eform_fold_{args.fold}.json.gz"))
