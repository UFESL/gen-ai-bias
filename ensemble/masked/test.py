import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", type=str)
args = parser.parse_args()
print(args.o)