import os
import random

import numpy as np

from matbench.constants import CLF_KEY, REG_KEY

MB_TEST_RANDOM_SEED = 1001
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_TEST = os.environ.get("MB_FULL_TESTS", False)


def model_random(training_outputs, test_inputs, response_type):
    r = random.Random(MB_TEST_RANDOM_SEED)

    r = np.random.RandomState(MB_TEST_RANDOM_SEED)

    l = len(test_inputs)

    if response_type == CLF_KEY:
        return r.choice([True, False], size=l)

    # Regression: simply sample from random distribution bounded by max
    # and min training samples
    if response_type == REG_KEY:
        pred = r.uniform(max(training_outputs), min(training_outputs), size=l)
        return pred
