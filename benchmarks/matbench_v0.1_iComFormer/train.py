from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union

import ignite
import torch

from ignite.contrib.handlers import TensorboardLogger
try:
    from ignite.contrib.handlers.stores import EpochOutputStore
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from matformer import models
from matformer.data import get_train_val_loaders
from matformer.config import TrainingConfig
# from matformer.models.pyg_att import Matformer

from jarvis.db.jsonutils import dumpjson
import json
import pprint

import os

# import sys
# sys.path.append("/mnt/data/shared/congfu/CompCrystal/NewModel_27/matformer/")
# from scheduler import PolynomialLR


import types
import math
import torch
from torch import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from torch.optim import Optimizer

class LRScheduler:

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.verbose = verbose

        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, lr))
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch_str, group, lr))


    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class _enable_get_lr_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False      

class PolynomialLR(LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (int): The power of the polynomial. Default: 1.0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Assuming optimizer uses lr = 0.001 for all groups
        >>> # lr = 0.001     if epoch == 0
        >>> # lr = 0.00075   if epoch == 1
        >>> # lr = 0.00050   if epoch == 2
        >>> # lr = 0.00025   if epoch == 3
        >>> # lr = 0.0       if epoch >= 4
        >>> scheduler = PolynomialLR(self.opt, total_iters=4, power=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False, end_lr=1e-5, decay_steps=10):
        self.total_iters = total_iters
        self.power = power
        self.end_lr = end_lr
        self.decay_steps = decay_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # print(self.last_epoch)
        # print(self._step_count)

        # if not self._get_lr_called_within_step:
        #     warnings.warn("To get the last learning rate computed by the scheduler, "
        #                   "please use `get_last_lr()`.", UserWarning)

        step = min(self.last_epoch, self.decay_steps)
        return [
            (
                (base_lr - self.end_lr)  * (1.0 - step / self.decay_steps) ** self.power + self.end_lr
            )
            for base_lr in self.base_lrs
        ]
    
        # if not self._get_lr_called_within_step:
        #     warnings.warn("To get the last learning rate computed by the scheduler, "
        #                   "please use `get_last_lr()`.", UserWarning)

        # if self.last_epoch == 0 or self.last_epoch > self.total_iters:
        #     return [group["lr"] for group in self.optimizer.param_groups]

        # decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        # return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        
        print(self.last_epoch)
        step = min(self.last_epoch, self.decay_steps)
        return [
            (
                (base_lr - self.end_lr)  * (1.0 - self.last_epoch / self.decay_steps) ** self.power + self.end_lr
            )
            for base_lr in self.base_lrs
        ]
    
        # return [
        #     (
        #         base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
        #     )
        #     for base_lr in self.base_lrs
        # ]
    
    # def decayed_learning_rate(step):
    #     step = min(step, decay_steps)
    #     return ((initial_learning_rate - end_learning_rate) *
    #             (1 - step / decay_steps) ^ (power)
    #             ) + end_learning_rate

###########################################################################################


# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_dgl(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    train_val_test_loaders=[],
    test_only=False,
    use_save=True,
    mp_id_list=None,
    train_inputs=None,
    train_outputs=None,
    test_inputs=None,
    test_outputs=None,
    model_variant=None,
):
    """
    `config` should conform to matformer.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    print(config)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)
            print('error in converting to training config!')
    import os
    
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    checkpoint_dir = os.path.join(config.output_dir)
    deterministic = False
    classification = False
    print("config:")
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    pprint.pprint(tmp) 
    if config.classification_threshold is not None:
        classification = True
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    # import pdb; pdb.set_trace()
    line_graph = True
    if not train_val_test_loaders:
        # use input standardization for all real-valued feature sets
        (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
            mean_train,
            std_train,
        ) = get_train_val_loaders(
            dataset=config.dataset,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            line_graph=line_graph,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=config.filename,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            output_dir=config.output_dir,
            matrix_input=config.matrix_input,
            pyg_input=config.pyg_input,
            use_lattice=config.use_lattice,
            use_angle=config.use_angle,
            use_save=use_save,
            mp_id_list=mp_id_list,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        prepare_batch = train_val_test_loaders[3]
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    # define network, optimizer, scheduler
    if model_variant == 'matformerinvariant':
        from matformer.models.pyg_att import MatformerInvariant as Matformer
    elif model_variant == 'matformerequivariant':
        from matformer.models.pyg_att import MatformerEquivariant as Matformer
    _model = {
        "matformer" : Matformer,
    }
    if std_train is None:
        std_train = 1.0
        print('std train is none!')
    print('std train:', std_train)
    if model is None:
        net = _model.get(config.model.name)(config.model)
        print("config:")
        pprint.pprint(config.model.dict())
    else:
        net = model

    net.to(device)
    if config.distributed:
        import torch.distributed as dist
        import os

        def setup(rank, world_size):
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

            # initialize the process group
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

        def cleanup():
            dist.destroy_process_group()

        setup(2, 2)
        net = torch.nn.parallel.DistributedDataParallel(
            net
        )
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100000,
            gamma=0.96,
        )
    elif config.scheduler == "polynomial":
        steps_per_epoch = len(train_loader)
        num_steps = config.epochs * steps_per_epoch
        scheduler = PolynomialLR(
            optimizer,
            decay_steps=num_steps,
            end_lr=1e-5,
        )

    # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
    }
    criterion = criteria[config.criterion]
    # set up training engine and evaluators
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError() * std_train, "neg_mae": -1.0 * MeanAbsoluteError() * std_train}
    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=deterministic,
    )
    evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    train_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    if test_only:
        checkpoint_tmp = torch.load('/your_model_path.pt')
        to_load = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_tmp)
        net.eval()
        targets = []
        predictions = []
        import time
        t1 = time.time()
        with torch.no_grad():
            for dat in test_loader:
                g, lg, _, target = dat
                try:
                    out_data = net([g.to(device), lg.to(device), _.to(device)])
                    success_flag=1
                except: # just in case
                    print('error for this data')
                    print(g)
                    success_flag=0
                if success_flag > 0:
                    out_data = out_data.cpu().numpy().tolist()
                    target = target.cpu().numpy().flatten().tolist()
                    if len(target) == 1:
                        target = target[0]
                    targets.append(target)
                    predictions.append(out_data)
        t2 = time.time()
        f.close()
        from sklearn.metrics import mean_absolute_error
        targets = np.array(targets) * std_train
        predictions = np.array(predictions) * std_train
        print("Test MAE:", mean_absolute_error(targets, predictions))
        print("Total test time:", t2-t1)
        return mean_absolute_error(targets, predictions)
    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    # checkpoint_tmp = torch.load("/mnt/data/shared/congfu/CompCrystal/NewModel_27/matformer/scripts/matbench_mp_e_form_equivariant_max25_epoch500_lr1e-3_L1_fold1/checkpoint_299.pt")
    # to_load = {
    #     "model": net,
    #     "optimizer": optimizer,
    #     "lr_scheduler": scheduler,
    #     "trainer": trainer,
    # }
    # Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_tmp)
    # print('checkpoint.pt loaded')
    # print('current epoch:', trainer.state.epoch)
    # print('current optimizer:', optimizer)
    # print('current scheduler:', scheduler)

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
        # evaluate save
        to_save = {"model": net}
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=5,
            filename_prefix='best',
            score_name="neg_mae",
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)
    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})
        # pbar.attach(evaluator,output_transform=lambda x: {"mae": x})

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    if config.store_outputs:
        # in history["EOS"]
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        # train_evaluator.run(train_loader)
        # evaluator.run(val_loader)

        # tmetrics = train_evaluator.state.metrics
        # vmetrics = evaluator.state.metrics
        # for metric in metrics.keys():
        #     tm = tmetrics[metric]
        #     vm = vmetrics[metric]
        #     if metric == "roccurve":
        #         tm = [k.tolist() for k in tm]
        #         vm = [k.tolist() for k in vm]
        #     if isinstance(tm, torch.Tensor):
        #         tm = tm.cpu().numpy().tolist()
        #         vm = vm.cpu().numpy().tolist()

        #     history["train"][metric].append(tm)
        #     history["validation"][metric].append(vm)

        # train_evaluator.run(train_loader)
        evaluator.run(val_loader)

        vmetrics = evaluator.state.metrics
        for metric in metrics.keys():
            vm = vmetrics[metric]
            t_metric = metric
            if metric == "roccurve":
                vm = [k.tolist() for k in vm]
            if isinstance(vm, torch.Tensor):
                vm = vm.cpu().numpy().tolist()

            history["validation"][metric].append(vm)

        
        
        epoch_num = len(history["validation"][t_metric])
        if epoch_num % 20 == 0:
            train_evaluator.run(train_loader)
            tmetrics = train_evaluator.state.metrics
            for metric in metrics.keys():
                tm = tmetrics[metric]
                if metric == "roccurve":
                    tm = [k.tolist() for k in tm]
                if isinstance(tm, torch.Tensor):
                    tm = tm.cpu().numpy().tolist()

                history["train"][metric].append(tm)
        else:
            tmetrics = {}
            tmetrics['mae'] = -1


        # for metric in metrics.keys():
        #    history["train"][metric].append(tmetrics[metric])
        #    history["validation"][metric].append(vmetrics[metric])

        if config.store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history["validation"],
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history["train"],
            )
        if config.progress:
            pbar = ProgressBar()
            if not classification:
                pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
                pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")
            else:
                pbar.log_message(f"Train ROC AUC: {tmetrics['rocauc']:.4f}")
                pbar.log_message(f"Val ROC AUC: {vmetrics['rocauc']:.4f}")

    if config.n_early_stopping is not None:
        if classification:
            my_metrics = "accuracy"
        else:
            my_metrics = "neg_mae"

        def default_score_fn(engine):
            score = engine.state.metrics[my_metrics]
            return score

        es_handler = EarlyStopping(
            patience=config.n_early_stopping,
            score_function=default_score_fn,
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)
    # optionally log results to tensorboard
    if config.log_tensorboard:

        tb_logger = TensorboardLogger(
            log_dir=os.path.join(config.output_dir, "tb_logs", "test")
        )
        for tag, evaluator in [
            ("training", train_evaluator),
            ("validation", evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "mae"],
                global_step_transform=global_step_from_engine(trainer),
            )

    trainer.run(train_loader, max_epochs=config.epochs)

    if config.log_tensorboard:
        test_loss = evaluator.state.metrics["loss"]
        tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
        tb_logger.close()
    if config.write_predictions and classification:
        net.eval()
        f = open(
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                # out_data = torch.exp(out_data.cpu())
                top_p, top_class = torch.topk(torch.exp(out_data), k=1)
                target = int(target.cpu().numpy().flatten().tolist()[0])

                f.write("%s, %d, %d\n" % (id, (target), (top_class)))
                targets.append(target)
                predictions.append(
                    top_class.cpu().numpy().flatten().tolist()[0]
                )
        f.close()
        from sklearn.metrics import roc_auc_score

        print("predictions", predictions)
        print("targets", targets)
        print(
            "Test ROCAUC:",
            roc_auc_score(np.array(targets), np.array(predictions)),
        )

    if (
        config.write_predictions
        and not classification
        and config.model.output_features > 1
    ):
        net.eval()
        mem = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(open("sc.pkl", "rb"))
                    out_data = list(
                        sc.transform(np.array(out_data).reshape(1, -1))[0]
                    )  # [0][0]
                target = target.cpu().numpy().flatten().tolist()
                info = {}
                info["id"] = id
                info["target"] = target
                info["predictions"] = out_data
                mem.append(info)
        dumpjson(
            filename=os.path.join(
                config.output_dir, "multi_out_predictions.json"
            ),
            data=mem,
        )
    if (
        config.write_predictions
        and not classification
        and config.model.output_features == 1
    ):
        net.eval()
        f = open(
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            for dat in test_loader:
                g, lg, _, target = dat
                out_data = net([g.to(device), lg.to(device), lg.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                target = target.cpu().numpy().flatten().tolist()
                if len(target) == 1:
                    target = target[0]
                targets.append(target)
                predictions.append(out_data)
        f.close()
        from sklearn.metrics import mean_absolute_error

        print(
            "Test MAE:",
            mean_absolute_error(np.array(targets), np.array(predictions)) * std_train,
            "STD train:",
            std_train,
        )
        if config.store_outputs and not classification:
            x = []
            y = []
            for i in history["EOS"]:
                x.append(i[0].cpu().numpy().tolist())
                y.append(i[1].cpu().numpy().tolist())
            x = np.array(x, dtype="float").flatten()
            y = np.array(y, dtype="float").flatten()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_train_set.csv"
                ),
                "w",
            )
            # TODO: Add IDs
            f.write("target,prediction\n")
            for i, j in zip(x, y):
                f.write("%6f, %6f\n" % (j, i))
                line = str(i) + "," + str(j) + "\n"
                f.write(line)
            f.close()
    # return history
    return np.array(targets) * std_train + mean_train, np.array(predictions) * std_train + mean_train


