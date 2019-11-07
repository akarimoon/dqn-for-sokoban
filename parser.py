import argparse

def parse():
    parser = argparse.ArgumentParser(description='Q-learning IB9 Sokoban')

    parser.add_argument('-f', '--free', action='store_true', default=False)
    parser.add_argument('-t', '--train', action='store_true', default=True)
    parser.add_argument('-c', '--call_pretrained', default=None)

    return parser

parser = parse()
args = parser.parse_args()
print(args.free)
