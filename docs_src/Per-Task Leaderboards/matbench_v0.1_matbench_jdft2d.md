# matbench_v0.1 matbench_jdft2d

## Individual Task Leaderboard for `matbench_jdft2d`

_Leaderboard for an individual task. Algorithms shown here may include both general purpose and specialized algorithms (i.e., algorithms which are only valid for a subset of tasks in the benchmark._

### Leaderboard

| algorithm | mean mae | std mae | mean rmse | max max_error |
|------|------|------|------|------|
| [AMMExpress v2020](/Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020) | **39.8497** | 9.8835 | 106.5460 | 1552.9102 | 
| [Dummy](/Full%20Benchmark%20Data/matbench_v0.1_dummy) | **67.2851** | 10.1832 | 126.8446 | 1491.7993 | 


<iframe src="static/task_matbench_v0.1_matbench_jdft2d.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>

### Dataset info

##### Description

Matbench v0.1 test dataset for predicting exfoliation energies from crystal structure (computed with the OptB88vdW and TBmBJ functionals). Adapted from the JARVIS DFT database. For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication for more details.

Number of samples: 636

Task type: regression

Input type: structure

##### Dataset columns

- exfoliation_en: Target variable. Exfoliation energy (meV).
- structure: Pymatgen Structure of the material.


##### Dataset reference

 `2D Dataset discussed in:
High-throughput Identification and Characterization of Two dimensional Materials using Density functional theory Kamal Choudhary, Irina Kalish, Ryan Beams & Francesca Tavazza Scientific Reports volume 7, Article number: 5179 (2017)
Original 2D Data file sourced from:
choudhary, kamal; https://orcid.org/0000-0001-9737-8074 (2018): jdft_2d-7-7-2018.json. figshare. Dataset.`

### Metadata

```
{'bibtex_refs': ['@Article{Dunn2020,\n'
                 'author={Dunn, Alexander\n'
                 'and Wang, Qi\n'
                 'and Ganose, Alex\n'
                 'and Dopp, Daniel\n'
                 'and Jain, Anubhav},\n'
                 'title={Benchmarking materials property prediction methods: '
                 'the Matbench test set and Automatminer reference '
                 'algorithm},\n'
                 'journal={npj Computational Materials},\n'
                 'year={2020},\n'
                 'month={Sep},\n'
                 'day={15},\n'
                 'volume={6},\n'
                 'number={1},\n'
                 'pages={138},\n'
                 'abstract={We present a benchmark test suite and an automated '
                 'machine learning procedure for evaluating supervised machine '
                 'learning (ML) models for predicting properties of inorganic '
                 'bulk materials. The test suite, Matbench, is a set of '
                 '13{\\thinspace}ML tasks that range in size from 312 to 132k '
                 'samples and contain data from 10 density functional '
                 'theory-derived and experimental sources. Tasks include '
                 'predicting optical, thermal, electronic, thermodynamic, '
                 "tensile, and elastic properties given a material's "
                 'composition and/or crystal structure. The reference '
                 'algorithm, Automatminer, is a highly-extensible, fully '
                 'automated ML pipeline for predicting materials properties '
                 'from materials primitives (such as composition and crystal '
                 'structure) without user intervention or hyperparameter '
                 'tuning. We test Automatminer on the Matbench test suite and '
                 'compare its predictive power with state-of-the-art crystal '
                 'graph neural networks and a traditional descriptor-based '
                 'Random Forest model. We find Automatminer achieves the best '
                 'performance on 8 of 13 tasks in the benchmark. We also show '
                 'our test suite is capable of exposing predictive advantages '
                 'of each algorithm---namely, that crystal graph methods '
                 'appear to outperform traditional machine learning methods '
                 'given {\\textasciitilde}104 or greater data points. We '
                 'encourage evaluating materials ML algorithms on the Matbench '
                 'benchmark and comparing them against the latest version of '
                 'Automatminer.},\n'
                 'issn={2057-3960},\n'
                 'doi={10.1038/s41524-020-00406-3},\n'
                 'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                 '}\n',
                 '@Article{Choudhary2017,\n'
                 'author={Choudhary, Kamal\n'
                 'and Kalish, Irina\n'
                 'and Beams, Ryan\n'
                 'and Tavazza, Francesca},\n'
                 'title={High-throughput Identification and Characterization '
                 'of Two-dimensional Materials using Density functional '
                 'theory},\n'
                 'journal={Scientific Reports},\n'
                 'year={2017},\n'
                 'volume={7},\n'
                 'number={1},\n'
                 'pages={5179},\n'
                 'abstract={We introduce a simple criterion to identify '
                 'two-dimensional (2D) materials based on the comparison '
                 'between experimental lattice constants and lattice constants '
                 'mainly obtained from Materials-Project (MP) density '
                 'functional theory (DFT) calculation repository. '
                 'Specifically, if the relative difference between the two '
                 'lattice constants for a specific material is greater than or '
                 'equal to 5%, we predict them to be good candidates for 2D '
                 'materials. We have predicted at least 1356 such 2D '
                 'materials. For all the systems satisfying our criterion, we '
                 'manually create single layer systems and calculate their '
                 'energetics, structural, electronic, and elastic properties '
                 'for both the bulk and the single layer cases. Currently the '
                 'database consists of 1012 bulk and 430 single layer '
                 'materials, of which 371 systems are common to bulk and '
                 'single layer. The rest of calculations are underway. To '
                 'validate our criterion, we calculated the exfoliation energy '
                 'of the suggested layered materials, and we found that in '
                 '88.9% of the cases the currently accepted criterion for '
                 'exfoliation was satisfied. Also, using molybdenum telluride '
                 'as a test case, we performed X-ray diffraction and Raman '
                 'scattering experiments to benchmark our calculations and '
                 'understand their applicability and limitations. The data is '
                 'publicly available at the website '
                 'http://www.ctcms.nist.gov/{\t'
                 'extasciitilde}knc6/JVASP.html.},\n'
                 'issn={2045-2322},\n'
                 'doi={10.1038/s41598-017-05402-0},\n'
                 'url={https://doi.org/10.1038/s41598-017-05402-0}\n'
                 '}',
                 '@misc{choudhary__2018, title={jdft_2d-7-7-2018.json}, '
                 'url={https://figshare.com/articles/jdft_2d-7-7-2018_json/6815705/1}, '
                 'DOI={10.6084/m9.figshare.6815705.v1}, abstractNote={2D '
                 'materials}, publisher={figshare}, author={choudhary, kamal '
                 'and https://orcid.org/0000-0001-9737-8074}, year={2018}, '
                 'month={Jul}}'],
 'columns': {'exfoliation_en': 'Target variable. Exfoliation energy (meV).',
             'structure': 'Pymatgen Structure of the material.'},
 'description': 'Matbench v0.1 test dataset for predicting exfoliation '
                'energies from crystal structure (computed with the OptB88vdW '
                'and TBmBJ functionals). Adapted from the JARVIS DFT database. '
                'For benchmarking w/ nested cross validation, the order of the '
                'dataset must be identical to the retrieved data; refer to the '
                'Automatminer/Matbench publication for more details.',
 'file_type': 'json.gz',
 'hash': '26057dc4524e193e32abffb296ce819b58b6e11d1278cae329a2f97817a4eddf',
 'input_type': 'structure',
 'mad': 67.20200406491116,
 'n_samples': 636,
 'num_entries': 636,
 'reference': '2D Dataset discussed in:\n'
              'High-throughput Identification and Characterization of Two '
              'dimensional Materials using Density functional theory Kamal '
              'Choudhary, Irina Kalish, Ryan Beams & Francesca Tavazza '
              'Scientific Reports volume 7, Article number: 5179 (2017)\n'
              'Original 2D Data file sourced from:\n'
              'choudhary, kamal; https://orcid.org/0000-0001-9737-8074 (2018): '
              'jdft_2d-7-7-2018.json. figshare. Dataset.',
 'target': 'exfoliation_en',
 'task_type': 'regression',
 'unit': 'meV/atom',
 'url': 'https://ml.materialsproject.org/projects/matbench_jdft2d.json.gz'}
```

