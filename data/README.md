# Data directories
Contains instructions of where each dataset is stored, and what the stored data represent

```                                
├── data                        <- Project data
│   ├── external                    <- Data from third party sources.
│   ├── interim                     <- Intermediate data that has been transformed.
│   │   └── identity_by_week            <- Directory containing the displayed identity subcategories for users that had at least one identity shown (4,732,035)
│   │
│   ├── processed                   <- The final, canonical data sets for modeling.
│   └── raw                         <- The original, immutable data dump.
│       └── description_changes.tsv.gz  <- Data containing all unique profiles of 9,142,850 users from 2020/4/1 to 2021/5/1
│
└── README.md
```