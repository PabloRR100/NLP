
# Project Architecture

```

app
├──  chekckpoints
│    └── model1.pkl  - here's the default config file.
│
├──  data  
│    └── inputs
│           └── corpus
│           └── catalogs
│    └── results
│           └── clustering
│           └── keyphrases ...
│
├──  src
│   └── algorithms
│           └── clustering
│                   └── clust_config.yml
│                   └── clust_utils.py
│                   └── cluster.py
│           └── texterank ...
│   └── preprocessing
│           └── adhoc_proccesing.py
│           └── catalog.py
│   └── visualizations
│           └── clustering
│                   └── clustering_dashboard.py
└── tests
     ├── jejeje test... xD
```

