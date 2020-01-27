# Assignment 4 (NMT)
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

Create virtual environment
```bash
conda env create --file local_env.yml
```

Activate and deactivate environment
```bash
conda activate local_nmt
conda deactivate
```

Install necessary packages (On your VM)
```bash
pip install -r gpu_requirements.txt
```

Generate the required vocab files
```bash
sh run.sh vocab
```

Train model on your local machine
```bash
sh run.sh train_local
```


