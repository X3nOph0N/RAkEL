"""
command example:
```python
python main.py --config-file configuration.yml
```
"""

from argparse import ArgumentParser
from sklearn.datasets import load_iris
from model import RAKEL
from numpy import load as data_load
from yaml import load as config_load
from yaml import Loader
from os.path import join


def main(LPModel: str, dataPath: str, k: int, m: int, **kwargs):

    X_train = data_load(join(dataPath, 'X_train.npy'))
    X_test = data_load(join(dataPath, 'X_test.npy'))
    Y_train = data_load(join(dataPath, 'Y_train.npy'))
    Y_test = data_load(join(dataPath, 'Y_test.npy'))

    RAkELClassifier = RAKEL(LPModel, kwargs)
    RAkELClassifier.fit(X_train, Y_train, k, m)
    RAkELClassifier.eval(X_test, Y_test, kwargs)

    return


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--config-file', type=str,
                           default='configuration.yml')
    args = argparser.parse_args()
    with open(args.config_file, 'rt') as f:
        configs = config_load(f, Loader=Loader)
    main(**configs)
