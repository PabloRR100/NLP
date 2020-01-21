cols = ['count', 'job', 'source']
vas = [
    [2,   'sales',   'A'],
    [4,   'sales',   'B'],
    [6,   'sales',   'C'],
    [3,   'sales',   'D'],
    [7,   'sales',   'E'],
    [5,   'market',  'A'],
    [3,   'market',  'B'],
    [2,   'market',  'C'],
    [4,   'market',  'D'],
    [1,   'market',  'E']]

df = pd.DataFrame(vas, columns = cols)


df.sort_values(['job', 'source'], ascending=False).groupby('job').head(3)

