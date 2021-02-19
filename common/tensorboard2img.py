import matplotlib.pyplot as plt
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=False, default='./img.png')
parser.add_argument('--title', type=str, required=False, default='RL algorithm')
parser.add_argument('--xlabel', type=str, required=False, default='epochs')
parser.add_argument('--ylabel', type=str, required=False, default='reward')
args = parser.parse_args()

data = pd.read_csv(args.csv_path)

plt.plot(data['Step'], data['Value'])
plt.title(args.title)
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
plt.savefig(args.save_path, dpi=60)
plt.show()
