from argparse import ArgumentParser

FINALLINE = "-----------------Final-------------------"
STARTRUN = "rewire"
LOGLINE = "Loss"
AUCLINE = "AUC results"
NDCGLINE = "nDCG results"
TAULINE = "Kendall's Tau"
TASKLINE = "task"

def processLogLine(line):
    split = line.split()
    loss = split[3]
    trainAUC = split[6]
    valAUC = split[9]
    testAUC = split[12]
    nDCG = split[14]
    kendallTau = split[17]
    return loss, trainAUC, valAUC, testAUC, nDCG, kendallTau

def main():
    parser = ArgumentParser()
    # general
    parser.add_argument('--filename', dest='filename', required=True, type=str,
                        help='file to analyze')
    parser.add_argument('--prefix', dest='prefix', required=True, type=str,
                        help='prefix for output line')
    args = parser.parse_args()

    with open(args.filename) as file:
        lines = file.readlines()
        losses = []
        testAUCs = []
        overFit = False
        printResult=False
        lastAUC = None
        lastNDCG = None
        lastTau = None
        lastTask = None
        for line in lines:
            if TASKLINE in line:
                if printResult:
                    print(args.prefix + " " + lastTask + " AUC: {0} {1}, NDCG: {2} {3}, Kendall Tau: {4} {5}, overfit: {6}".format(lastAUC[0], lastAUC[1], lastNDCG[0], lastNDCG[1], lastTau[0], lastTau[1], overFit))
                    printResult = False
                splitLine = line.split()
                lastTask = splitLine[1]
            elif STARTRUN in line:
                if len(losses) > 1:
                    if testAUC[0] < testAUC[-1]:
                        overFit = True
                loss = []
                testAUC = []
                lastNDCG = None
                lastAUC = None
                lastTau = None
            elif LOGLINE in line:
                loss, trainAUC, valAUC, testAUC, nDCG, kendallTau = processLogLine(line)
                losses.append(testAUCs)
                testAUCs.append(testAUC)
            elif FINALLINE in line:
                printResult = True
            elif TAULINE in line:
                splitLine = line.split()
                lastTau = (splitLine[2], splitLine[3])
            elif NDCGLINE in line:
                splitLine = line.split()
                lastNDCG = (splitLine[2], splitLine[3])
            elif AUCLINE in line:
                splitLine = line.split()
                lastAUC = (splitLine[2], splitLine[3])

        if printResult:
            print(args.prefix + " " + lastTask + " AUC: {0} {1}, NDCG: {2} {3}, Kendall Tau: {4} {5}, overfit: {6}".format(lastAUC[0], lastAUC[1], lastNDCG[0], lastNDCG[1], lastTau[0], lastTau[1], overFit))



if __name__ == '__main__':
	main()
