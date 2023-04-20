# Data directories
Contains instructions of where each dataset is stored, and what the stored data represent

```                                
├── data                        <- Project data
│   ├── external                    <- Data from third party sources.
│   ├── interim                     <- Intermediate data that has been transformed.
│   │   ├── all_uids                        <- Contains data of all user ids that were used in the initial 15M sample of users active during both 2020.04 and 2021.04
│   │   ├── description_changes             <- Directory containing the changes made to filter down description changes and get extracted identities
│   │   │   ├── remove-heavy-users          <- Removed users who were in top-5 percentile for either number of followers, posts, or had verified status
│   │   │   ├── remove-non-english          <- Removed users whose language in profile contained non-english
│   │   │   ├── filtered                    <- Filtered version where all users to remove were removed
│   │   │   └── extracted                   <- Contains results obtained after running identity extraction algorithms
│   │   │
│   │   ├── identity_by_week                <- Directory containing the displayed identity subcategories for users that had at least one identity shown (4,732,035)
│   │   ├── identity_classifier-train_data  <- Directory used for creating the training datasets for the identity classifier
│   │   │
│   │   ├── propensity-score-matching       <- Directory containing the files that contain covariate for propensity matching
│   │   │
│   │   └── treated-control-users           <- Directory containing the treated and (potential) control users for propensity matching
│   │
│   ├── processed                   <- The final, canonical data sets for modeling or data analysis.
│   │   └── identity_classifier-train_data  <- Directory containing the training datasets for the identity classifier
│   │
│   └── raw                         <- The original, immutable data dump.
│       ├── user_info
│       │   └── (obsolete) user_profile-2020.04.json.gz     <- File containing the user info of all 8.9M/9.1M users tracked initially, can be used for matching
│       │
│       ├── identity_classifier-train_data      <- Directory containing the raw tweets obtained from the 3-month (2020.04-06) activities of users mapped to pos/neg samples
│       │
│       ├── treated-control-tweets              <- Tweets of 1.6-year history from the 1.9M treated + potential control users that went through all the filtering
│       │   ├── activity_around_profile_update      <- The tweets that have been created ~8 weeks prior to and ~12 weeks post treatment
│       │   │   ├── activities_made                     <- All activities where our user of interest is the person who posted or responed to another's post
│       │   │   └── activities_origin                   <- All activities where our user of interest is the person who was targeted by another user via retweets/quote/reply/mention
│       │   │
│       │   ├── tweets                              <- All tweets of the 1.6-year history, 1.9M users
│       │   ├── users                               <- All user objects of the 1.6-year history, 1.9M users
│       │   └── all_user_profiles.json.gz           <- The earliest instance of the user object for the 1.9M users
│       │
│       └── description_changes        <- Directory containing profile updates of 15M users who were active in 2020/4/1 and 2021/4/1
│           ├── description_changes_0_changes.tsv.gz        <- Data containing all unique profiles of 9,142,850 users from 2020/4/1 to 2021/5/1
│           └── description_changes_1plus_changes.tsv.gz    <- Data containing all unique profiles of 9,142,850 users from 2020/4/1 to 2021/5/1
│
└── README.md
```