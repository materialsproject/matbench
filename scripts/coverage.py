"""
for use with the coverage job in this repo's github workflows
"""

import os

from monty.serialization import loadfn


fname = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir), "coverage.json")
coverage = loadfn(fname)

total_percentage = coverage["totals"]["percent_covered"]

if total_percentage < 75.0:
    raise ValueError(f"Coverage of {total_percentage} is less than threshold, 75%.")