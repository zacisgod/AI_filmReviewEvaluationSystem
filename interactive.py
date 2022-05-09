"""
File: interactive.py
Name: 
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""
from submission import extractWordFeatures
from util import interactivePrompt

WEIGHTS_PATH = ('weights')
weights = {}
def main():
	with open(WEIGHTS_PATH) as f:
		for line in f:
			key, val = line.split('	', 1)
			weights[str(key)] = float(val)
	interactivePrompt(extractWordFeatures, weights)


if __name__ == '__main__':
	main()