# Data

The experiments use the public **SAM 40** EEG dataset. The data are not redistributed here.

> Ghosh, R., Deb, N., Sengupta, K., et al. *SAM 40: Dataset of 40 subject EEG recordings to monitor the induced-stress while performing Stroop color-word test, arithmetic task, and mirror image recognition task.* Data in Brief 40 (2022), 107772. [doi:10.1016/j.dib.2021.107772](https://doi.org/10.1016/j.dib.2021.107772) ([open access](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8749216/))

## Folder layout the code expects

```
<data-root>/
├── Relax/          Relax_sub_1.csv       ... Relax_sub_40.csv
├── Arithmetic/     Arithmetic_sub_1.csv  ... Arithmetic_sub_40.csv
├── Mirror_image/   Mirror_image_sub_1.csv ... Mirror_image_sub_40.csv
└── Stroop/         Stroop_sub_1.csv      ... Stroop_sub_40.csv
```

Each CSV has 9,600 rows and 33 columns: a leading index column (`0`-`3199`, repeated three times) and 32 numeric columns.

## What a row is

Every row is one time sample of 32 EEG channels:

- the index runs 0 to 3,199 three times per file; 3,200 samples is 25 s at 128 Hz, and the dataset records three 25 s trials per task;
- consecutive rows are strongly autocorrelated (lag-1 correlation about 0.999 on the channels checked), and the amplitude jumps where one trial ends and the next begins;
- all values are finite; ranges are roughly ±2,000 across the 160 files.

The classifiers therefore see one 32-value vector per time sample. There is no windowing or hand-built feature step in the experiment code: the index column is dropped, the 32 channel values are standardised on the training fold, and the models are fitted on them. In total each condition has 40 files x 9,600 rows = 384,000 rows, and each binary task has 768,000 rows (384,000 per class).

Which exact export of the dataset these CSVs came from (and any filtering already applied to it) is not recorded in this repository.
