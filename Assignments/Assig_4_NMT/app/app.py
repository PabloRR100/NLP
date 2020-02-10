
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=True)

# import torch
# import torch.nn as nn
# from nmt_model import NMT 

# # x = input('Español: ', )
# x = 'Me gustan los libros'

# # Load Model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = torch.load('model.bin', map_location=torch.device(device))
# print('Inside the model: [{}]'.format(model.keys()))

# net = NMT(
#     embed_size=model['args']['embed_size'],
#     hidden_size=model['args']['hidden_size'],
#     vocab=model['vocab'],
#     dropout_rate=model['args']['dropout_rate'])

# net.load_state_dict(model['state_dict'])
# net.eval()

# x = x.split(' ')
# net.beam_search(x)[0].value

# # Direct Approach Using the Class
# net2 = NMT.load('model.bin')
# net2.eval()
# net2.beam_search(x)[0].value



