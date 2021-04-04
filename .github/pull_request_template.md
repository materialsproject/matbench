# Matbench Pull Request Template

Thanks for making a PR to Matbench! We appreciate your contribution (like, a lot). To make things run smoothly, check out the following templates, 
depending on what kind of PR you are making.

If you are making a benchmark submission (i.e., you have tried an algorithm on Matbench and want to appear on the leaderboard), 
please use the template under [Benchmark submissions](#benchmark-submissions).

If you are making changes to the core matbench code, data, or docs, please use the template under [Core code/data/docs changes](#core-codedatadocs-changes).




## Benchmark submissions

Benchmark submissions can include a full benchmark on any of the benchmarks Matbench submits, as well as any subset of tasks within a benchmark (e.g., 3 Matbench v0.1 tasks your algorithm supports).

### Brief description of your algorithm

You should a brief description of your algorithm in the pull request body. This can include any details you'd like.


### Included files

If you are making a benchmark submission, please **only** include the submission as a folder in the `/benchmarks` directory with the format `<benchmark_name>_<algorithm_name>`. Your PR should have no other changes to the core code.
The submission should have these three required files, as indicated in the
[docs](https://hackingmaterials.lbl.gov/matbench/running_and_submitting_benchmarks/):


Example
```
-- benchmarks
---- matbench_v0.1_my_algorithm
------ results.json.gz             # required filename
------ notebook.ipynb              # required filename
------ info.json                   # required filename
```


**Please make sure each of these files has the information specified in the [docs](https://hackingmaterials.lbl.gov/matbench/running_and_submitting_benchmarks/).**

If you have other short/small files required for the notebook, please give a brief overview of what each one is used for and how to use it.


## Core code/data/docs changes


### Brief description of changes

Please include a brief description of the changes you are making, in bullet point format.



### Tests

Indicate if your code requires new tests and whether they are included with your PR. ALL core code/data/docs changes adding new features **must** have new tests for them.



### Closed issues or PRs

Indicate if your PR closes any currently open issues or supercedes any other currently open PRs.