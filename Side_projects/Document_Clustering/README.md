
# Project Architecture

```

app
├──  chekckpoints
│    └── kmeans_isocynate_Arturito_2020_10_20.pkl
│    └── hdbscan_isocynate_Arturito_2020_10_20.pkl
│
├──  data  
│    └── inputs
│           └── corpus
│                   └── corpus_only_english
│                   └── corpus_heavy_preproccessing
│           └── catalogs
│                   └── catalog_only_english
│                   └── catalgo_heavy_preproccessing
│    └── results
│           └── clustering
│                   └── clustering_isocynate_Arturito_2020_01_20.pkl
│                   └── clustering_isocynate_Arturito_2010_01_22.pkl
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

