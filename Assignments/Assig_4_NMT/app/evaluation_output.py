import os
from typing import Dict

import torch
from nmt_model import NMT
from run import beam_search
from utils import read_corpus


def check_dir_exists(directory_path):
    """
    Check if directory exists, if not create it

    :param directory_path: path to the directory
    :return:
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def decode(args: Dict[str, str]):
    """ Performs decoding on the autograder test set
    Make sure to run this code before submitting the code to the auto`grader
    @param args (Dict): args from cmd line
    """

    test_data_src = read_corpus(args['SOURCE_FILE'], source='src')
    model = NMT.load(args['MODEL_PATH'])

    if args['CUDA']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['BEAM_SIZE']),
                             max_decoding_time_step=int(args['MAX_DECODING_TIME_STEP']))

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = {
        'SOURCE_FILE': './en_es_data/grader.es',
        'OUTPUT_FILE': './outputs/gradescope_test_outputs.txt',
        'MODEL_PATH': './model.bin',
        'CUDA': True,
        'MAX_DECODING_TIME_STEP': 70,
        'BEAM_SIZE': 5
    }
    check_dir_exists('./outputs/')
    decode(args)


if __name__ == '__main__':
    main()
