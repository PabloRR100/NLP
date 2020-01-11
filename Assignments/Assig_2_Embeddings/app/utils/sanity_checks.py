import numpy as np
import random
from utils.utils import normalizeRows


def dummy():
    random.seed(31415)
    np.random.seed(9265)

    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    return dataset, dummy_vectors, dummy_tokens


inputs = {
    'test_word2vec': {
        'currentCenterWord': "c",
        'windowSize': 3,
        'outsideWords': ["a", "b", "e", "d", "b", "c"]
    },
    'test_naivesoftmax': {
        'centerWordVec': np.array([-0.27323645, 0.12538062, 0.95374082]).astype(float),
        'outsideWordIdx': 3,
        'outsideVectors': np.array([[-0.6831809, -0.04200519, 0.72904007],
                                    [0.18289107, 0.76098587, -0.62245591],
                                    [-0.61517874, 0.5147624, -0.59713884],
                                    [-0.33867074, -0.80966534, -0.47931635],
                                    [-0.52629529, -0.78190408, 0.33412466]]).astype(float)

    },
    'test_sigmoid': {
        'x': np.array([-0.46612273, -0.87671855, 0.54822123, -0.36443576, -0.87671855, 0.33688521
                          , -0.87671855, 0.33688521, -0.36443576, -0.36443576, 0.54822123]).astype(float)
    }
}

outputs = {
    'test_word2vec': {
        'loss': 11.16610900153398,
        'dj_dv': np.array(
            [[0., 0., 0.],
             [0., 0., 0.],
             [-1.26947339, -1.36873189, 2.45158957],
             [0., 0., 0.],
             [0., 0., 0.]]).astype(float),
        'dj_du': np.array(
            [[-0.41045956, 0.18834851, 1.43272264],
             [0.38202831, -0.17530219, -1.33348241],
             [0.07009355, -0.03216399, -0.24466386],
             [0.09472154, -0.04346509, -0.33062865],
             [-0.13638384, 0.06258276, 0.47605228]]).astype(float)

    },
    'test_naivesoftmax': {
        'loss': 2.217424877675181,
        'dj_dvc': np.array([-0.17249875, 0.64873661, 0.67821423]).astype(float),
        'dj_du': np.array([[-0.11394933, 0.05228819, 0.39774391],
                           [-0.02740743, 0.01257651, 0.09566654],
                           [-0.03385715, 0.01553611, 0.11817949],
                           [0.24348396, -0.11172803, -0.84988879],
                           [-0.06827005, 0.03132723, 0.23829885]]).astype(float)
    },
    'test_sigmoid': {
        's': np.array(
            [0.38553435, 0.29385824, 0.63372281, 0.40988622, 0.29385824, 0.5834337, 0.29385824, 0.5834337, 0.40988622,
             0.40988622, 0.63372281]).astype(float),
    }

}

sample_vectors_expected = {
    "female": [
        0.6029723815239835,
        0.16789318536724746,
        0.22520087305967568,
        -0.2887330648792561,
        -0.914615719505456,
        -0.2206997036383445,
        0.2238454978107194,
        -0.27169214724889107,
        0.6634932978039564,
        0.2320323110106518
    ],
    "cool": [
        0.5641256072125872,
        0.13722982658305444,
        0.2082364803517175,
        -0.2929695723456364,
        -0.8704480862547578,
        -0.18822962799771015,
        0.24239616047158674,
        -0.29410091959922546,
        0.6979644655991716,
        0.2147529764765611
    ]
}
