from ax.core import SearchSpace
from ax.core.parameter_constraint import SumConstraint, OrderConstraint
from ax import RangeParameter, ChoiceParameter, ParameterType

# %% constraint parameters and constraints
betas1 = RangeParameter(
    name="betas1", parameter_type=ParameterType.FLOAT, lower=0.5, upper=0.9999
)
betas2 = RangeParameter(
    name="betas2", parameter_type=ParameterType.FLOAT, lower=0.5, upper=0.9999
)
emb_scaler = RangeParameter(
    name="emb_scaler", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
)
pos_scaler = RangeParameter(
    name="pos_scaler", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
)
order_constraint = OrderConstraint(lower_parameter=betas1, upper_parameter=betas2)
sum_constraint = SumConstraint(
    parameters=[emb_scaler, pos_scaler], is_upper_bound=True, bound=1.0
)
parameter_constraints = [order_constraint, sum_constraint]

# %% search space
search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="batch_size", parameter_type=ParameterType.INT, lower=32, upper=256
        ),
        RangeParameter(
            name="fudge", parameter_type=ParameterType.FLOAT, lower=0.0, upper=0.1
        ),
        RangeParameter(
            name="d_model", parameter_type=ParameterType.INT, lower=100, upper=1024
        ),
        RangeParameter(name="N", parameter_type=ParameterType.INT, lower=1, upper=10),
        RangeParameter(
            name="heads", parameter_type=ParameterType.INT, lower=1, upper=10
        ),
        RangeParameter(
            name="out_hidden4", parameter_type=ParameterType.INT, lower=32, upper=512
        ),
        emb_scaler,
        pos_scaler,
        ChoiceParameter(
            name="bias", parameter_type=ParameterType.BOOL, values=[False, True]
        ),
        RangeParameter(
            name="dim_feedforward",
            parameter_type=ParameterType.INT,
            lower=1024,
            upper=4096,
        ),
        RangeParameter(
            name="dropout", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        ),
        ChoiceParameter(
            name="elem_prop",
            parameter_type=ParameterType.STRING,
            values=["mat2vec", "magpie", "onehot"],
        ),
        RangeParameter(
            name="epochs_step", parameter_type=ParameterType.INT, lower=5, upper=20
        ),
        RangeParameter(
            name="pe_resolution",
            parameter_type=ParameterType.INT,
            lower=2500,
            upper=10000,
        ),
        RangeParameter(
            name="ple_resolution",
            parameter_type=ParameterType.INT,
            lower=2500,
            upper=10000,
        ),
        ChoiceParameter(
            name="criterion",
            parameter_type=ParameterType.STRING,
            values=["RobustL1", "RobustL2"],
        ),
        RangeParameter(
            name="lr", parameter_type=ParameterType.FLOAT, lower=0.0001, upper=0.006
        ),
        betas1,
        betas2,
        RangeParameter(
            name="eps",
            parameter_type=ParameterType.FLOAT,
            lower=0.0000001,
            upper=0.0001,
        ),
        RangeParameter(
            name="weight_decay",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="alpha", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0,
        ),
        RangeParameter(name="k", parameter_type=ParameterType.INT, lower=2, upper=10),
    ],
    parameter_constraints=parameter_constraints,
)