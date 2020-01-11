
Install virtualenv
```
sudo pip install virtualenv 
```

Create and activate virtual environment (using virtualenv)
```
virtualenv venv -p python3.6 (python 3.6/3.5 would be preferrable)
source venv/bin/activate
```

Create and activate virtual environment (using conda)
```
conda env create --file local_env.yml
conda activate A3
conda deactivate
```

Unzip data folder
```bash
unzip data.zip
```

install pytorch with no cuda
```bash
check here: https://pytorch.org/
```

Install all dependencies (only required if you setup virtual environment via virtualenv)
```
pip install -r requirements.txt
```


Assignment submission
```
# zip the assignment submission folder
sh collect_submission.sh
cd ..
```

