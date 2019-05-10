from argparse import ArgumentParser
from tqdm import tqdm
from typing import Iterable

import json
import os


def tag_dataset(dataset: Iterable[str], key: str, tag: str) -> str:
    for line in tqdm(dataset):
        example = json.loads(line.strip())
        example[args.key] = dataset_name
        yield json.dumps(example, ensure_ascii=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('datasets', nargs='+',
                        help='input MATERIAL annotations files')
    parser.add_argument('-k', '--key', dest='key', default='dataset',
                        help='the json key for the dataset name')
    args = parser.parse_args()

    for file in args.datasets:
        dataset_name = os.path.splitext(os.path.basename(file))[0]
        with open(file) as fp:
            for ex in tag_dataset(fp, args.key, dataset_name):
                print(ex)
