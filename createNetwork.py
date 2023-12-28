import numpy as np
import pandas as pd
from digitRec import DigitRecogniser, getPredictions, getAccuracy
from digitRec2 import DigitRecogniser2
import sys, getopt

def main(argv) -> None:

    opts, args = getopt.getopt(argv, "d: l: i: a:")
    if len(opts) < 2:
        raise Exception("Not enough arguments")
    for opt, arg in opts:

        if opt == '-d':
            if arg.lower() == "mnist":
                print("Using MNIST Dataset")
                filePath = 'data/train.csv'
            elif arg.lower() == "custom":
                print("Using custom dataset")
                filePath = 'data/custom.csv'
            else:
                raise Exception("Dataset not found")
    
        elif opt == '-l':
            if arg == '1':
                numberOfLayers = 1
            elif arg == '2':
                numberOfLayers = 2
            else:
                raise Exception("Invalid number of layers")
            
        elif opt == '-i':
            try:
                iterations = int(arg)
            except:
                raise ValueError("Invalid iterations input")
        
        elif opt == "-a":
            try:
                alpha = float(arg)
            except:
                raise ValueError("Invalid learning rate value")
            
    layer1Nodes = int(input("\nEnter number of nodes for Hidden Layer 1: "))
    if numberOfLayers == 2:
        layer2Nodes = int(input("\nEnter number of nodes for Hidden Layer 2: "))
        network = DigitRecogniser2(filePath, layer1Nodes, layer2Nodes, iterations, alpha)
    else:
        network = DigitRecogniser(filePath, layer1Nodes, iterations, alpha)
    network.gradientDesc()

    data = np.array(pd.read_csv('data/test.csv'))
    data = data.T
    testLabels = data[0]
    testInputs = data[1:]/255
    _, _, _, output = network.forwardProp(testInputs)
    print(f'Accuracy on Custom Test Set: {getAccuracy(getPredictions(output), testLabels)}')

if __name__ == "__main__":
    main(sys.argv[1:])