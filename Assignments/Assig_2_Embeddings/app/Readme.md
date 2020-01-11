Install virtualenv
```
sudo pip install virtualenv 
```

Create and activate virtual environment
```
virtualenv venv -p python3.5
source venv/bin/activate
```

Install all requirements
```
pip install -r requirements.txt
```

Train word2vec and generate files *sampleVectors.json* and *word_vectors.png*

**Note: Do not change the hyperparameter values in run.py script**  
```
python run.py
```

Sanity check on sampleVectors.json
```
python test_sample_vectors.py
```

Assignment submission(for students)
```
# zip the assignment submission folder
sh collect_submission.sh
```
