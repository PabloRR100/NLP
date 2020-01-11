- Install virtualenv
```
sudo pip install virtualenv 
```

- Create and activate virtual environment
```
virtualenv venv -p python3.5
source venv/bin/activate
```

- Install all the requirements
```
pip install -r requirements.txt
```

- Perform sanity checks on `computeCoOccurrenceMatrix`, `distinctWords` and `reduceToKDim`
```
python co_occurence.py
```

- Compute `co-occurrence`, run `SVD` and create `co_occurence_embeddings.png`
```
python run.py
```

- Assignment submission
```
# zip the assignment submission folder
sh collect_submission.sh
```


- Miscellaneous
If this error occurs:
```
ImportError: No named '_tkinter', please install the python3-tk package
```
Install `tkinter`
```
sudo apt-get install python3-tk
```
