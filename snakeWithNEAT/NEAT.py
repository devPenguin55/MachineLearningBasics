from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import defaultdict
from copy import deepcopy as copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import pickle as pkl
import random as r
import traceback
import snake
import math
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "100,100"
import pygame
pygame.init()
import sys

# r.seed(555)

trainModelInstruction = 'train' in sys.argv[1]
if trainModelInstruction:
    trainModelGenerationsInstruction = int(sys.argv[2])
    trainModelVisualizationGenerationStartInstruction = int(sys.argv[3])


WIDTH, HEIGHT = (500, 500)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake')
# r.seed(6) # 1, 60, 61, 62

GLOBAL_INNOVATION_NUMBERS = {}
GLOBAL_CURRENT_INNOVATION_NUMBER = 0

# inputs -> (facing up) (facing down) (facing left) (facing right) (food ahead) (food left) (food right) (obstacle ahead) (obstacle left) (obstacle right)
# outputs -> (face up) (face down) (face left) (face right)
GLOBAL_NODE_GENE_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [11, 12, 13, 14] # input nodes and output nodes
GLOBAL_CURRENT_NODE_GENE_LABEL = len(GLOBAL_NODE_GENE_LABELS)

GLOBAL_SPLIT_RESULTS = {}

class NodeGene:
    def __init__(self, label, geneType):
        self.label = label
        self.geneType = geneType
    
    def __eq__(self, other):
        return (
            isinstance(other, NodeGene) and
            self.label == other.label
        )

class ConnectionGene:
    def __init__(self, inLabel, outLabel, weight, state, innovationNumber):
        self.inLabel = inLabel
        self.outLabel = outLabel
        self.weight = weight
        self.state = state
        self.innovationNumber = innovationNumber
    
    def __eq__(self, other):
        return (
            isinstance(other, ConnectionGene) and
            self.inLabel == other.inLabel and
            self.outLabel == other.outLabel and
            self.weight == other.weight and
            self.state == other.state and
            self.innovationNumber == other.innovationNumber
        )

geneTypes = ['input']*10+['output']*4
nodeGenes = [NodeGene(i, geneTypes[i-1]) for i in range(1, GLOBAL_CURRENT_NODE_GENE_LABEL+1)]

class Network:
    def __init__(self, nodeGenes:list[NodeGene], connectionGenes:list[ConnectionGene]=[], networkId=None):
        global GLOBAL_CURRENT_INNOVATION_NUMBER, GLOBAL_INNOVATION_NUMBERS, GLOBAL_CURRENT_NODE_GENE_LABEL, GLOBAL_NODE_GENE_LABELS, GLOBAL_SPLIT_RESULTS
        self.nodeGenes = copy(nodeGenes)
        self.connectionGenes = copy(connectionGenes)

        # print([i.label for i in self.nodeGenes])
        self.inputGenes = [i for i in self.nodeGenes if i.label in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        self.outputGenes = [i for i in self.nodeGenes if i.label in [11, 12, 13, 14]]

        self.networkId = networkId

    def registerInnovationNumber(self, inLabel, outLabel):
        global GLOBAL_CURRENT_INNOVATION_NUMBER, GLOBAL_INNOVATION_NUMBERS, GLOBAL_CURRENT_NODE_GENE_LABEL, GLOBAL_NODE_GENE_LABELS, GLOBAL_SPLIT_RESULTS
        
        key = f'{inLabel}->{outLabel}'
        if key not in GLOBAL_INNOVATION_NUMBERS:
            GLOBAL_INNOVATION_NUMBERS[key] = GLOBAL_CURRENT_INNOVATION_NUMBER + 1
            GLOBAL_CURRENT_INNOVATION_NUMBER += 1
        return key

    def mutateAddConnection(self, forceInLabel=None, forceOutLabel=None, forceState=None):
        global GLOBAL_CURRENT_INNOVATION_NUMBER, GLOBAL_INNOVATION_NUMBERS, GLOBAL_CURRENT_NODE_GENE_LABEL, GLOBAL_NODE_GENE_LABELS, GLOBAL_SPLIT_RESULTS

        # find a random source and target gene to create a connection between
        if forceInLabel is None and forceOutLabel is None:
            inLabel = r.choice(self.inputGenes).label
            outLabel = r.choice(self.outputGenes).label
        else:
            inLabel = forceInLabel
            outLabel = forceOutLabel

        for connection in self.connectionGenes:
            if connection.inLabel == inLabel and connection.outLabel == outLabel:
                return

        key = self.registerInnovationNumber(inLabel, outLabel)
        self.connectionGenes.append(ConnectionGene(inLabel, outLabel, r.uniform(-2, 2), r.choice([True, False]) if forceState is None else forceState, GLOBAL_INNOVATION_NUMBERS[key]))

    def mutateAddNode(self):
        global GLOBAL_CURRENT_INNOVATION_NUMBER, GLOBAL_INNOVATION_NUMBERS, GLOBAL_CURRENT_NODE_GENE_LABEL, GLOBAL_NODE_GENE_LABELS, GLOBAL_SPLIT_RESULTS

        # first find a connection to modify
        if not self.connectionGenes:
            raise Exception('No Connection Genes Created Yet To Modify')
        oldConnectionGene = r.choice(self.connectionGenes)
        #      after must remove it from the available connection genes because it will be split
        self.connectionGenes.remove(oldConnectionGene)


        # check if this connection has already been split, if so then use the resulting node gene
        key = f'{oldConnectionGene.inLabel}->{oldConnectionGene.outLabel}'
        if key in GLOBAL_SPLIT_RESULTS:
            newNodeGene = GLOBAL_SPLIT_RESULTS[key]
            self.nodeGenes.append(newNodeGene)
        else:
            # create the node gene
            GLOBAL_CURRENT_NODE_GENE_LABEL += 1
            newNodeGene = NodeGene(GLOBAL_CURRENT_NODE_GENE_LABEL, 'hidden')
            self.nodeGenes.append(newNodeGene)
            GLOBAL_NODE_GENE_LABELS.append(GLOBAL_CURRENT_NODE_GENE_LABEL)

            # store the split
            GLOBAL_SPLIT_RESULTS[key] = newNodeGene

        # next, modify the connections around it
        #     first, the old connection's input to the new gene
        key = self.registerInnovationNumber(oldConnectionGene.inLabel, newNodeGene.label)
        self.connectionGenes.append(ConnectionGene(oldConnectionGene.inLabel, newNodeGene.label, 1, oldConnectionGene.state, GLOBAL_INNOVATION_NUMBERS[key]))
        #     second, the new gene to the old connection's output
        key = self.registerInnovationNumber(newNodeGene.label, oldConnectionGene.outLabel)
        self.connectionGenes.append(ConnectionGene(newNodeGene.label, oldConnectionGene.outLabel, r.uniform(-2, 2), oldConnectionGene.state, GLOBAL_INNOVATION_NUMBERS[key]))

    def mutateEnableDisable(self, enableProbability=0.4, disableProbability=0.4):
        global GLOBAL_CURRENT_INNOVATION_NUMBER, GLOBAL_INNOVATION_NUMBERS, GLOBAL_CURRENT_NODE_GENE_LABEL, GLOBAL_NODE_GENE_LABELS, GLOBAL_SPLIT_RESULTS
        
        if not self.connectionGenes:
            raise Exception('No Connection Genes Created Yet To Modify')
        
        for connectionIdx in range(len(self.connectionGenes)):
            enabled = self.connectionGenes[connectionIdx].state
            if not enabled and r.random() < enableProbability:
                self.connectionGenes[connectionIdx].state = True
            elif enabled and r.random() < disableProbability:
                self.connectionGenes[connectionIdx].state = False
    
    def mutateWeightShift(self, shiftProbability=0.5):
        global GLOBAL_CURRENT_INNOVATION_NUMBER, GLOBAL_INNOVATION_NUMBERS, GLOBAL_CURRENT_NODE_GENE_LABEL, GLOBAL_NODE_GENE_LABELS, GLOBAL_SPLIT_RESULTS
        
        if not self.connectionGenes:
            raise Exception('No Connection Genes Created Yet To Modify')
        
        for connectionIdx in range(len(self.connectionGenes)):
            if r.random() < shiftProbability:
                self.connectionGenes[connectionIdx].weight += r.uniform(-0.1, 0.1)

    def mutateWeightRandom(self, randomWeightProbability=0.25):
        global GLOBAL_CURRENT_INNOVATION_NUMBER, GLOBAL_INNOVATION_NUMBERS, GLOBAL_CURRENT_NODE_GENE_LABEL, GLOBAL_NODE_GENE_LABELS, GLOBAL_SPLIT_RESULTS
        
        if not self.connectionGenes:
            raise Exception('No Connection Genes Created Yet To Modify')
        
        for connectionIdx in range(len(self.connectionGenes)):
            if r.random() < randomWeightProbability:
                self.connectionGenes[connectionIdx].weight = r.uniform(-2, 2)

    def feedForward(self, inputs):
        inputs = [int(i) for i in inputs]

        connectionGenes = [(i.inLabel, i.outLabel, i.state, i.weight) for i in self.connectionGenes if i.state]
        inputLabels = [i.label for i in self.inputGenes]
        outputLabels = [i.label for i in self.outputGenes]
        labelToNode = {i.label: i for i in self.nodeGenes}

        # Step 1: Prune connections using DFS from outputs
        usefulConnections = set()
        visited = set()

        def dfs(nodeLabel):
            if nodeLabel in visited:
                return
            visited.add(nodeLabel)
            for conn in connectionGenes:
                if conn[1] == nodeLabel:
                    usefulConnections.add(conn)
                    dfs(conn[0])

        for outLabel in outputLabels:
            dfs(outLabel)

        usefulConnections = list(usefulConnections)

        # Step 2: Build graph
        inMap = {}  # how many inputs each node expects
        outMap = {}  # what outputs each node feeds into
        for inLabel, outLabel, _, _ in usefulConnections:
            if outLabel not in inMap:
                inMap[outLabel] = 0
            inMap[outLabel] += 1
            if inLabel not in outMap:
                outMap[inLabel] = []
            outMap[inLabel].append((outLabel, _))  # _ is weight

        # Step 3: Initialize node values
        nodeValues = {}
        for label in inputLabels:
            nodeValues[label] = inputs[inputLabels.index(label)]

        readyQueue = inputLabels[:]
        seenConnections = set()

        # Step 4: Feedforward loop
        while readyQueue:
            current = readyQueue.pop(0)
            if current not in outMap:
                continue
            for outLabel, weight in outMap[current]:
                connKey = (current, outLabel)
                if connKey in seenConnections:
                    continue
                seenConnections.add(connKey)

                val = nodeValues.get(current, 0) * weight
                if outLabel not in nodeValues:
                    nodeValues[outLabel] = 0
                nodeValues[outLabel] += val

                inMap[outLabel] -= 1
                if inMap[outLabel] == 0:
                    # All inputs received, apply activation
                    nodeValues[outLabel] = 1 / (1 + math.exp(-nodeValues[outLabel]))
                    readyQueue.append(outLabel)

        # Step 5: Extract output values
        networkOutput = []
        for outLabel in outputLabels:
            val = nodeValues.get(outLabel, 0)
            networkOutput.append((outLabel, val))

        # Step 6: Decide output
        networkOutput.sort(key=lambda x: x[0])
        outputVals = [val for _, val in networkOutput]
        maxVal = max(outputVals)
        candidateIndices = [i for i, v in enumerate(outputVals) if v == maxVal]

        if len(candidateIndices) == len(outputLabels):
            return 0
        return r.choice(candidateIndices)

    # def feedForward(self, inputs):
    #     inputs = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    #     inputs = [int(i) for i in inputs]

    #     # first, find the meaningful nodes that actually influence the output nodes
    #     connectionGenes = [(i.inLabel, i.outLabel, i.state, i.weight) for i in self.connectionGenes]
    #     inputLabels = [i.label for i in self.inputGenes]
    #     outputLabels = [i.label for i in self.outputGenes]

    #     verifiedPaths = []

    #     unidentifiedLinkageNodes = copy(self.outputGenes)
    #     unidentifiedLinkageNodes = [(i, []) for i in unidentifiedLinkageNodes]


    #     labelToNode = {i.label:i for i in self.nodeGenes}
    #     while True:
    #         nextPhase = []
    #         allConnectionOutLabels = [i[1] for i in connectionGenes]
    #         for outputNode, curPath in unidentifiedLinkageNodes:
    #             # see if any connections connect to an output node
    #             for idx, outLabel in enumerate(allConnectionOutLabels):
    #                 # if the label matches and the connection is enabled
    #                 if outputNode.label == outLabel and connectionGenes[idx][2]:
    #                     # if reached, then will continue to next phase if not part of the inputs
    #                     if connectionGenes[idx][0] in inputLabels:
    #                         # reached the inputs, means that the path of connections is valid!
    #                         verifiedPaths.append([connectionGenes[idx]]+curPath)
    #                     else:
    #                         matchedNode = copy(labelToNode[connectionGenes[idx][0]])
    #                         nextPhase.append((matchedNode, [connectionGenes[idx]]+curPath))
    #         unidentifiedLinkageNodes = copy(nextPhase)
    #         if not nextPhase:
    #             break
    #     # print('\n')
    #     # print(verifiedPaths)

    #     # alreadySeenConnections = set()
    #     # for idx in range(len(verifiedPaths)):
    #     #     path = verifiedPaths[idx]
    #     #     finalPath = [path[0]]
    #     #     if len(path) > 1:
    #     #         for conn in path[1:]:
    #     #             if conn not in alreadySeenConnections:
    #     #                 finalPath.append(conn)
    #     #             alreadySeenConnections.add(conn)
    #     #     if not (len(finalPath) == 1 and len(path) > 1):
    #     #         verifiedPaths[idx] = finalPath
    #     #     else:
    #     #         verifiedPaths[idx] = []
        
    #     # finalVerifiedPaths = []
    #     # for path in verifiedPaths:
    #     #     if path:
    #     #         finalVerifiedPaths.append(path)
        
    #     # verifiedPaths = finalVerifiedPaths
    #     # print('\n')
    #     for e in verifiedPaths:
    #         print(e)

    #     # now, find the expected amount of incoming data to each node
    #     nodeExpectedIncomingConnections = {}
    #     nodeStoredData = {}
    #     inputNodes = []

    #     alreadySeenConnections = set()
    #     for path in verifiedPaths:
    #         inputNode = path[0][0]
    #         inputNodes.append([inputNode, inputs[inputLabels.index(inputNode)]])

    #         for connection in path:
    #             if connection in alreadySeenConnections:
    #                 continue
    #             alreadySeenConnections.add(connection)
    #             outputNode = connection[1]
    #             nodeStoredData[outputNode] = []
    #             if nodeExpectedIncomingConnections.get(outputNode):
    #                 nodeExpectedIncomingConnections[outputNode] += 1
    #             else:
    #                 nodeExpectedIncomingConnections[outputNode] = 1  
    #     inputNodes = [list(j) for j in list(set([tuple(i) for i in inputNodes]))]
    #     print(nodeExpectedIncomingConnections)
    #     print(nodeStoredData)
    #     print(sorted(inputNodes, key=lambda x:x[0]), len(inputNodes))
    #     # quit()

    #     # actual feedforward process

    #     def getContiguousSubsets(arr):
    #         subsets = []
    #         for start in range(len(arr)):
    #             for end in range(start + 1, len(arr) + 1):
    #                 subsets.append(arr[start:end])
    #         return subsets

    #     # each node js like deposits to a sum of values for a node, then when it hits the right count it activation functions it
    #     # then also the output nodes are js the 
    #     networkOutput = []

    #     totalCurrentConnectionsFound = []
    #     while any(nodeExpectedIncomingConnections.values()):
    #         outputNodes = []
    #         for inputNode, inputValue in inputNodes:
    #             connectionFound = None

    #             # within verified paths i need to tell it to not deposit values in certain repeat sequences
    #             # so maybe the inputs amount should be as many as there are connections but the thing should
    #             # be that for the repeating ones there are only that many minus the repeating ones
    #             # 
    #             for connectionPath in verifiedPaths:
    #                 for connectionIdx, connection in enumerate(connectionPath):
    #                     connectionPathUpto = connectionPath[:connectionIdx+1]
                        
                        
    #                     if connection[0] == inputNode and ((not totalCurrentConnectionsFound) or all([(([connectionPathUpto] if len(connectionPathUpto) == 1 else connectionPathUpto) not in getContiguousSubsets(k)) for k in totalCurrentConnectionsFound])):
    #                         connectionFound = copy(connection)
    #                         # if len(connectionPathUpto) == 1:
    #                             # connectionPathUpto = [connectionPathUpto]

    #                         # ! TODO -> write smt to see if a list is a subset of another list efficiently
    #                         # ! now 16 deposits 2 and 17 gets 2 deposited into it
    #                         # if inputNode == 16:
    #                         #     print('hello', connectionPathUpto, 'is the registered path')
    #                         #     for j in totalCurrentConnectionsFound:
    #                         #         print(j, connectionPathUpto, [connectionPathUpto] not in getContiguousSubsets(j))
    #                         #     print([([connectionPathUpto] not in getContiguousSubsets(k)) for k in totalCurrentConnectionsFound])
    #                         #     print(all([([connectionPathUpto] not in getContiguousSubsets(k)) for k in totalCurrentConnectionsFound]))
    #                         #     pass
    #                         print('hello', connectionFound, [connectionPathUpto] if len(connectionPathUpto) == 1 else connectionPathUpto)
    #                         totalCurrentConnectionsFound.append([connectionPathUpto] if len(connectionPathUpto) == 1 else connectionPathUpto)
    #                         # break
    #                         outputNodeValue = inputValue * connectionFound[3]
    #                         nodeStoredData[connectionFound[1]].append(outputNodeValue)
    #                         nodeExpectedIncomingConnections[connectionFound[1]] -= 1
    #                 # if connectionFound is not None:
    #                     # break
                

    #         print()
    #         print(nodeStoredData)
    #         print(nodeExpectedIncomingConnections)
    #         # quit()

    #         for node in list(nodeExpectedIncomingConnections.keys()):
    #             if nodeExpectedIncomingConnections[node] == 0:
    #                 # node has no more data to expect, run activation function on it
    #                 newNodeValue = 1 / ( 1 + math.e ** (-sum(nodeStoredData[node])))
    #                 print(node, 'finished')
    #                 if node in outputLabels:
    #                     networkOutput.append((node, newNodeValue))
    #                 else:
    #                     outputNodes.append((node, newNodeValue))

    #                 del nodeExpectedIncomingConnections[node]
    #                 del nodeStoredData[node]

            
    #         print()
    #         print(outputNodes)
    #         print(nodeStoredData)
    #         print(nodeExpectedIncomingConnections)
    #         print('outputnodes len', len(outputNodes))

    #         inputNodes = outputNodes[:]
    #         if len(outputNodes) == 0:
    #             quit()
    #         # quit()

    #     # print('DONE')
    #     for node in outputLabels:
    #         if node not in [i[0] for i in networkOutput]:
    #             networkOutput.append((node, 0))
    #     networkOutput.sort(key=lambda x: x[0])
    #     networkOutputJustValues = [i[1] for i in networkOutput]
    #     maxVal = max(networkOutputJustValues)
    #     candidateIndices = [i for i, val in enumerate(networkOutputJustValues) if val == maxVal]
    #     if len(candidateIndices) == len(outputLabels):
    #         return 0
    #     decision = r.choice(candidateIndices)
    #     print(networkOutput, decision)
    #     return decision

def crossoverTwoNetworks(net1:Network, net2:Network, net1IsMostFitNet):
    net1ConnectionGenes = net1.connectionGenes
    net2ConnectionGenes = net2.connectionGenes

    # find the overlapping connection genes and take a random one
    finalNetConnectionGenes = []
    
    net1ConnectionGenes.sort(key=lambda x: x.innovationNumber)
    net2ConnectionGenes.sort(key=lambda x: x.innovationNumber)
    i, j = 0, 0
    while i < len(net1ConnectionGenes) and j < len(net2ConnectionGenes):
        curConnectionGeneNet1 = copy(net1ConnectionGenes[i])
        curConnectionGeneNet2 = copy(net2ConnectionGenes[j])

        if curConnectionGeneNet1.innovationNumber == curConnectionGeneNet2.innovationNumber:
            # matching gene
            finalNetConnectionGenes.append(r.choice([curConnectionGeneNet1, curConnectionGeneNet2]))
            i += 1
            j += 1
        elif curConnectionGeneNet1.innovationNumber < curConnectionGeneNet2.innovationNumber:
            # disjoint from net 1
            if net1IsMostFitNet:
                finalNetConnectionGenes.append(curConnectionGeneNet1)
            i += 1
        elif curConnectionGeneNet2.innovationNumber < curConnectionGeneNet1.innovationNumber:
            # disjoint from net 2
            if not net1IsMostFitNet:
                finalNetConnectionGenes.append(curConnectionGeneNet2)
            j += 1

    if net1IsMostFitNet:    
        while i < len(net1ConnectionGenes):
            # excess from net 1
            finalNetConnectionGenes.append(net1ConnectionGenes[i])
            i += 1
    else:
        while j < len(net2ConnectionGenes):
            # excess from net 2
            finalNetConnectionGenes.append(net2ConnectionGenes[j])
            j += 1

    inputLabels = [n.label for n in nodeGenes[:10]]
    outputLabels = [n.label for n in nodeGenes[-4:]]

    # derive node genes
    derivedNodeGenes = set()
    for connectionGene in finalNetConnectionGenes:
        if connectionGene.inLabel not in inputLabels:
            derivedNodeGenes.add((connectionGene.inLabel))
        if connectionGene.outLabel not in outputLabels:
            derivedNodeGenes.add((connectionGene.outLabel))

    # print([i for i in derivedNodeGenes], inputLabels, outputLabels)
    derivedNodeGenes = list(derivedNodeGenes) + inputLabels + outputLabels
    derivedNodeGenes.sort()
    derivedNodeGenes = [
        NodeGene(label, 'input' if label in inputLabels else 'output' if label in outputLabels else 'hidden')
        for label in derivedNodeGenes
    ]

    return Network(derivedNodeGenes, finalNetConnectionGenes)

class Game:
    def __init__(self, net):
        self.net = net
        self.gameOngoing = True
        self.snake = snake.Snake()
        self.apple = snake.Apple()
        self.duration = 0
        self.foodEaten = 0

        self.maxDurationBeforeFoodEaten = 90
        self.idleDuration = 0

    def gameStep(self):
        netInputs = [
            self.snake.dir == (0, 1),  # (facing up)
            self.snake.dir == (0, -1), # (facing down)
            self.snake.dir == (-1, 0), # (facing left)
            self.snake.dir == (1, 0)   # (facing right)
        ]

        # food ahead
        if self.snake.dir == (0, 1):
            netInputs.append(self.apple.y > self.snake.body[0][1])
        elif self.snake.dir == (0, -1):
            netInputs.append(self.apple.y < self.snake.body[0][1])
        elif self.snake.dir == (-1, 0):
            netInputs.append(self.apple.x < self.snake.body[0][0])
        elif self.snake.dir == (1, 0):
            netInputs.append(self.apple.x > self.snake.body[0][0])

        # food to left
        if self.snake.dir == (0, 1):
            netInputs.append(self.apple.x < self.snake.body[0][0])
        elif self.snake.dir == (0, -1):
            netInputs.append(self.apple.x > self.snake.body[0][0])
        elif self.snake.dir == (-1, 0):
            netInputs.append(self.apple.y < self.snake.body[0][1])
        elif self.snake.dir == (1, 0):
            netInputs.append(self.apple.y > self.snake.body[0][1])
            

        # food to right
        if self.snake.dir == (0, 1):
            netInputs.append(self.apple.x > self.snake.body[0][0])
        elif self.snake.dir == (0, -1):
            netInputs.append(self.apple.x < self.snake.body[0][0])
        elif self.snake.dir == (-1, 0):
            netInputs.append(self.apple.y > self.snake.body[0][1])
        elif self.snake.dir == (1, 0):
            netInputs.append(self.apple.y < self.snake.body[0][1])

        # obstacle ahead
        if self.snake.dir == (0, 1):
            netInputs.append(self.snake.body[0][1] == 500-self.snake.stepSize or (self.snake.body[0][0], self.snake.body[0][1]+self.snake.stepSize) in self.snake.body)
        elif self.snake.dir == (0, -1):
            netInputs.append(self.snake.body[0][1] == self.snake.stepSize or (self.snake.body[0][0], self.snake.body[0][1]-self.snake.stepSize) in self.snake.body)
        elif self.snake.dir == (-1, 0):
            netInputs.append(self.snake.body[0][0] == self.snake.stepSize or (self.snake.body[0][0]-self.snake.stepSize, self.snake.body[0][1]) in self.snake.body)
        elif self.snake.dir == (1, 0):
            netInputs.append(self.snake.body[0][0] == 500-self.snake.stepSize or (self.snake.body[0][0]+self.snake.stepSize, self.snake.body[0][1]) in self.snake.body)

        # obstacle to left
        if self.snake.dir == (0, 1):
            netInputs.append(self.snake.body[0][0] == self.snake.stepSize or (self.snake.body[0][0]-self.snake.stepSize, self.snake.body[0][1]) in self.snake.body)
        elif self.snake.dir == (0, -1):
            netInputs.append(self.snake.body[0][0] == 500-self.snake.stepSize or (self.snake.body[0][0]+self.snake.stepSize, self.snake.body[0][1]) in self.snake.body)
        elif self.snake.dir == (-1, 0):
            netInputs.append(self.snake.body[0][1] == self.snake.stepSize or (self.snake.body[0][0], self.snake.body[0][1]-self.snake.stepSize) in self.snake.body)
        elif self.snake.dir == (1, 0):
            netInputs.append(self.snake.body[0][1] == 500-self.snake.stepSize or (self.snake.body[0][0], self.snake.body[0][1]+self.snake.stepSize) in self.snake.body)
            
        # obstacle to right
        if self.snake.dir == (0, 1):
            netInputs.append(self.snake.body[0][0] == 500-self.snake.stepSize or (self.snake.body[0][0]+self.snake.stepSize, self.snake.body[0][1]) in self.snake.body)
        elif self.snake.dir == (0, -1):
            netInputs.append(self.snake.body[0][0] == self.snake.stepSize or (self.snake.body[0][0]-self.snake.stepSize, self.snake.body[0][1]) in self.snake.body)
        elif self.snake.dir == (-1, 0):
            netInputs.append(self.snake.body[0][1] == 500-self.snake.stepSize or (self.snake.body[0][0], self.snake.body[0][1]+self.snake.stepSize) in self.snake.body)
        elif self.snake.dir == (1, 0):
            netInputs.append(self.snake.body[0][1] == self.snake.stepSize or (self.snake.body[0][0], self.snake.body[0][1]-self.snake.stepSize) in self.snake.body)


        # print(netInputs)
        # outputs -> (face up) (face down) (face left) (face right)

 
        dirChoices = {
            0:(0, 1),
            1:(0, -1),
            2:(-1, 0),
            3:(1, 0)
        }


        oldDir = self.snake.dir
        newDir = dirChoices.get(self.net.feedForward(netInputs))
        if newDir == (-oldDir[0], -oldDir[1]):
            # self.gameOngoing = False
            # return
            pass
        else:
            self.snake.changeDir(newDir)
        
        gameOver = self.snake.step()
        if gameOver:
            self.gameOngoing = False
            return 
        
        wasFoodEatenInGameStep = self.apple.step(self.snake)
        self.foodEaten += wasFoodEatenInGameStep

        if wasFoodEatenInGameStep:
            self.snake.addBodySegment()
            self.idleDuration = 0
        else:
            self.idleDuration += 1
        
        if self.idleDuration > self.maxDurationBeforeFoodEaten:
            self.gameOngoing = False
            self.duration -= self.maxDurationBeforeFoodEaten
            return 

        self.duration += 1

    def fitness(self):
        return (self.duration-self.idleDuration*0)//3 + self.foodEaten * 100

    def simulateFullGame(self):
        while self.gameOngoing:
            self.gameStep()
        return self.fitness()

def drawGenome(genome, ax=None):
    g = nx.DiGraph()
    labelToType = {}
    connectionMap = defaultdict(list)

    # Add all nodes to the graph
    for node in genome.nodeGenes:
        labelToType[node.label] = node.geneType
        g.add_node(node.label)

    # Collect all connections (enabled or not)
    for conn in genome.connectionGenes:
        connectionMap[conn.outLabel].append(conn.inLabel)

    # Step 1: Compute depth (layer) of each node
    nodeDepth = {}

    def computeDepth(node, visiting=set()):
        if node in nodeDepth:
            return nodeDepth[node]
        if node in visiting:
            # Cycle detected, assign depth 0 to break the cycle
            return 0
        visiting.add(node)

        preds = connectionMap[node]
        nodeType = labelToType[node]

        if nodeType == 'input':
            nodeDepth[node] = 0
        elif preds:
            nodeDepth[node] = 1 + max(computeDepth(p, visiting) for p in preds)
        elif nodeType == 'output':
            # If output node has no preds, place it after max layer
            maxLayer = max(nodeDepth.values()) if nodeDepth else 0
            nodeDepth[node] = maxLayer + 1
        else:
            nodeDepth[node] = 0

        visiting.remove(node)
        return nodeDepth[node]

    for node in g.nodes:
        computeDepth(node)

    # Align all outputs to the max output depth layer
    outputNodes = [node for node, t in labelToType.items() if t == 'output']
    maxOutputDepth = max(nodeDepth.get(node, 0) for node in outputNodes) if outputNodes else 0
    for node in outputNodes:
        nodeDepth[node] = maxOutputDepth

    # Step 2: Assign (x, y) positions based on layer
    layers = defaultdict(list)
    for node, depth in nodeDepth.items():
        layers[depth].append(node)

    pos = {}
    for depth, nodes in layers.items():
        spacingY = 1.0 / (len(nodes) + 1)
        for i, node in enumerate(nodes):
            pos[node] = (depth, 1 - (i + 1) * spacingY)

    # Step 3: Assign colors to nodes
    colorMap = {
        'input': 'skyblue',
        'output': 'lightgreen',
        'hidden': 'lightgray'
    }
    nodeColors = [colorMap.get(labelToType[n], 'gray') for n in g.nodes]

    # Step 4: Add only enabled connections for main edges
    for conn in genome.connectionGenes:
        if conn.state:
            # g.add_edge(conn.inLabel, conn.outLabel, weight=conn.innovationNumber)
            g.add_edge(conn.inLabel, conn.outLabel, weight=round(conn.weight, 2))

    # Optional: Draw disabled connections as dashed gray lines
    disabledEdges = [
        (conn.inLabel, conn.outLabel, round(conn.weight, 2))
        for conn in genome.connectionGenes if not conn.state
    ]

    edgeLabels = nx.get_edge_attributes(g, 'weight')

    # Step 5: Draw graph
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))


    nx.draw(
        g, pos,
        ax=ax,
        with_labels=True,
        node_color=nodeColors,
        edge_color='black',
        arrows=True,
        node_size=501,
        edgecolors='black',
        linewidths=1.5,
        font_weight='bold'
    )

    # Edge weights
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edgeLabels, ax=ax)

    # Draw disabled connections (optional)
    for u, v, w in disabledEdges:
        if u in pos and v in pos:
            nx.draw_networkx_edges(
                g, pos,
                edgelist=[(u, v)],
                style='dashed',
                edge_color='gray',
                arrows=True, ax=ax
            )
    ax.axis('off')
    
def createInitialStateOfNetwork(net):
    for curIn in r.sample(list(range(10)), r.randint(1, 10)):
        for curOut in r.sample(list(range(4)), r.randint(1, 4)):
            net.mutateAddConnection(1+curIn, 11+curOut, True)
    return net
        
def findFitnessOfNetworks(networks, graphics=False, fig=None, axs=None):
    output = []
    games = [[Game(network), gameIdx] for gameIdx, network in enumerate(networks)]
    gameColors = [(r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)) for _ in networks]
    currentGameFitnessesForGraphics = [[i, 0] for i in range(len(networks))]
    while len(output) != len(networks):
        if graphics:
            screen.fill((50, 50, 50))
        
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass

            topNetwork = max(currentGameFitnessesForGraphics, key=lambda x: x[1]*games[x[0]][0].gameOngoing)

            # axs.clear()
            # axs.axis('off')
            # fig.patch.set_alpha(0.0)
            # axs.patch.set_alpha(0.0)
            # drawGenome(games[topNetwork[0]][0].net, ax=axs)
            # plt.tight_layout()
            # canvas.draw()
            # renderer = canvas.get_renderer()
            # rawData = renderer.buffer_rgba()
            # size = canvas.get_width_height()
            # surf = pygame.image.frombuffer(rawData, size, "RGBA")
            # scaledSurf = pygame.transform.smoothscale(surf, (1.33*400, 400))  # New pixel size
            # screen.blit(scaledSurf, (screen.get_width() - 1.33*400, 10))

        for game, gameIdx in games:
            #############################
            if game.gameOngoing:
                game.gameStep()
                if graphics:
                    currentGameFitnessesForGraphics[gameIdx][1] = game.fitness()
                    if currentGameFitnessesForGraphics[gameIdx][0] == topNetwork[0]:
                        for segment in game.snake.body:
                            if segment == game.snake.body[0]:  
                                segmentColor = gameColors[gameIdx]
                            else:
                                segmentColor = [
                                    max(gameColors[gameIdx][0]-50, 0),
                                    max(gameColors[gameIdx][1]-50, 0),
                                    max(gameColors[gameIdx][2]-50, 0)
                                ]
                            pygame.draw.rect(screen, segmentColor, (segment[0], segment[1], game.snake.stepSize, game.snake.stepSize)) # x, y, width, height
                        pygame.draw.rect(screen, segmentColor, (game.apple.x, game.apple.y, game.apple.stepSize, game.apple.stepSize)) # x, y, width, height
                if not game.gameOngoing:
                    output.append([game.net, game.fitness(), gameIdx])
        if graphics:
            pygame.display.flip()
            pygame.time.delay(1)
    output.sort(key=lambda x: x[2])
    output = [[i[0], i[1]] for i in output]
    return output, fig, axs

class Species:
    def __init__(self, individuals, speciesId):
        self.individuals = individuals
        self.speciesId = speciesId
        self.originalAmountOfIndividuals = len(self.individuals)
    
    def addIndividual(self, individual):
        self.individuals.append(individual)
        self.originalAmountOfIndividuals += 1

    def getRepresentative(self):
        return self.individuals[0]
    
    def eliminateBottomOfSpecies(self, threshold=0.5):
        self.individuals.sort(key=lambda x: x[1])
        amtOfIndividualsToEliminate = math.ceil(len(self.individuals) * threshold)


        if amtOfIndividualsToEliminate >= len(self.individuals):
            self.individuals = []
            return True # species went extinct
        
        self.individuals = self.individuals[amtOfIndividualsToEliminate:]
        return False
        
    def performFitnessSharing(self):
        for individualIndex in range(len(self.individuals)):
            self.individuals[individualIndex][1] /= len(self.individuals)

def selection(fitnesses:list[list[Network, int]], existingSpecies=[], c1=1.0, c2=1.0, c3=0.4, threshold=1.5):
    # speciation

    newlyCreatedSpeciesIdCounter = 0

    for network, fitness in fitnesses:
        matchedWithSpecies = False
        if existingSpecies:
            for speciesIndex, species in enumerate(existingSpecies):
                # find the genetic distance from the network to the species
                representativeNetwork, representativeFitness = species.getRepresentative()

                N = max(len(network.connectionGenes), len(representativeNetwork.connectionGenes)) # normalizing factor
                D = 0 # will compute later, is the amount of disjoint genes
                E = 0 # will compute later, is the amount of excess genes
                W = [] # will compute later, is the average weight difference of matching genes


                # next, find the matching, disjoint, and excess amount of genes
                net1ConnectionGenes = network.connectionGenes
                net2ConnectionGenes = representativeNetwork.connectionGenes
    
                net1ConnectionGenes.sort(key=lambda x: x.innovationNumber)
                net2ConnectionGenes.sort(key=lambda x: x.innovationNumber)
                i, j = 0, 0
                while i < len(net1ConnectionGenes) and j < len(net2ConnectionGenes):
                    curConnectionGeneNet1 = net1ConnectionGenes[i]
                    curConnectionGeneNet2 = net2ConnectionGenes[j]

                    if curConnectionGeneNet1.innovationNumber == curConnectionGeneNet2.innovationNumber:
                        # matching gene
                        W.append(abs(curConnectionGeneNet1.weight - curConnectionGeneNet2.weight))
                        i += 1
                        j += 1
                    elif curConnectionGeneNet1.innovationNumber < curConnectionGeneNet2.innovationNumber:
                        # disjoint from net 1
                        D += 1
                        i += 1
                    elif curConnectionGeneNet2.innovationNumber < curConnectionGeneNet1.innovationNumber:
                        # disjoint from net 2
                        D += 1
                        j += 1

    
                while i < len(net1ConnectionGenes):
                    # excess from net 1
                    E += 1
                    i += 1
                    
                while j < len(net2ConnectionGenes):
                    # excess from net 2
                    E += 1
                    j += 1
                
                if W:
                    W = sum(W) / len(W)
                else:
                    W = 0

                if N == 0:
                    N = 1

                geneticDistance = ((c1 * E) / N)  +  ((c2 * D) / N)  +  (c3 * W)
                
                if geneticDistance < threshold:
                    existingSpecies[speciesIndex].addIndividual([network, fitness])
                    matchedWithSpecies = True
                    break

        if not matchedWithSpecies:
            existingSpecies.append(Species([[network, fitness]], newlyCreatedSpeciesIdCounter))

    # print(f'Speciation Concluded, {len(existingSpecies)} Species Exist.')

    # print(sorted([i[1] for i in existingSpecies[0].individuals]))
    # print(len(existingSpecies[0].individuals))

    # s = 0
    # for sp in existingSpecies:
    #     s += sp.originalAmountOfIndividuals
    # print(s)

    survivingSpecies = []
    amountOfPopulationIncreases = 0
    for speciesIndex in range(len(existingSpecies)):
        speciesExtinct = existingSpecies[speciesIndex].eliminateBottomOfSpecies(threshold=0.5)
        if not speciesExtinct:
            survivingSpecies.append(existingSpecies[speciesIndex])
        else:
            amountOfPopulationIncreases += 1

    # s = 0
    # for sp in survivingSpecies:
    #     s += sp.originalAmountOfIndividuals
    # print(s)  
    # print(amountOfPopulationIncreases)

    while amountOfPopulationIncreases:
        curSpecies = r.choice(survivingSpecies)
        curSpecies.originalAmountOfIndividuals += 1
        amountOfPopulationIncreases -= 1


    # s = 0
    # for sp in survivingSpecies:
    #     s += sp.originalAmountOfIndividuals
    # print(s)
    # print(sorted([i[1] for i in survivingSpecies[0].individuals]))
    # print(f'Elimination Of Bottom Of Species Concluded, {len(survivingSpecies)} Species Remain.')

    for species in survivingSpecies:
        species.performFitnessSharing()

    # print(sorted([round(i[1], 2) for i in survivingSpecies[0].individuals]))
    # print('Species Fitness Sharing Completed.')

    # selection
    totalSpeciesFromIndividualSpeciesSelection = []
    for speciesIdx in range(len(survivingSpecies)):
        species = list(survivingSpecies[speciesIdx].individuals)
        species.sort(key=lambda x: x[1], reverse=True)
        newSpecies = []

        newSpeciesTargetAmountOfIndividuals = survivingSpecies[speciesIdx].originalAmountOfIndividuals
        
        # make half of the target fulfilled
        for individualIndex in range(len(species)):
            curIndividualNetwork = species[individualIndex][0]
            if r.uniform(0, 1) < 0.25:
                mutateProb = r.uniform(0, 1)
                if mutateProb > 0.95:
                    curIndividualNetwork.mutateWeightRandom()
                elif mutateProb > 0.75:
                    curIndividualNetwork.mutateEnableDisable()
                elif mutateProb > 0.5:
                    curIndividualNetwork.mutateWeightShift()
                elif mutateProb > 0.48:
                    curIndividualNetwork.mutateAddConnection()
                elif mutateProb > 0.47:
                    curIndividualNetwork.mutateAddNode()
            newSpecies.append(copy(curIndividualNetwork))

        # make the last half of the target fulfilled
        for parent1Index in range(len(species)):
            for parent2Index in range(len(species)):
                if parent1Index != parent2Index and len(newSpecies) < newSpeciesTargetAmountOfIndividuals:
                    newSpecies.append(crossoverTwoNetworks(species[parent1Index][0], species[parent2Index][0], species[parent1Index][1]>species[parent2Index][1]))

        while len(newSpecies) < newSpeciesTargetAmountOfIndividuals:
            newIndividual = r.choice(newSpecies)
            mutationToPerform = r.choice([
                newIndividual.mutateAddConnection,
                newIndividual.mutateAddNode,
                newIndividual.mutateEnableDisable, 
                newIndividual.mutateWeightRandom, 
                newIndividual.mutateWeightShift
            ])
            mutationToPerform()
            newSpecies.append(newIndividual)

        if newSpecies:
            totalSpeciesFromIndividualSpeciesSelection.append(newSpecies)

    # print(f'Selection And Reproduction Completed, {len(totalSpeciesFromIndividualSpeciesSelection)} Species Exist.')

    return totalSpeciesFromIndividualSpeciesSelection

def flattenSpecies(totalSpeciesFromIndividualSpeciesSelection):
    totalSpeciesFromIndividualSpeciesSelection = totalSpeciesFromIndividualSpeciesSelection
    output = []
    amtOfSpecies = 0
    for species in totalSpeciesFromIndividualSpeciesSelection:
        if species:
            amtOfSpecies += 1
        for individual in species:
            output.append(individual)
    return amtOfSpecies, output

if trainModelInstruction:
    networks = [Network(nodeGenes, networkId=networkId) for networkId in range(100)]

    for netIndex in range(len(networks)):
        networks[netIndex] = copy(createInitialStateOfNetwork(networks[netIndex]))

    iterator = tqdm(range(trainModelGenerationsInstruction), colour='green')

    fig, axs = plt.subplots(1, 1, figsize=(14, 8))
    canvas = FigureCanvas(fig)

    for generation in iterator:
        fitnesses, fig, axs = findFitnessOfNetworks(networks, graphics=generation>trainModelVisualizationGenerationStartInstruction, fig=fig, axs=axs)
        # print(sorted([i[1] for i in fitnesses], reverse=True))
        bestIndividual = sorted(fitnesses, key=lambda x: x[1])[-1][0]

        amtOfSpecies, networks = flattenSpecies(selection(fitnesses, []))
        iterator.set_postfix({
            'Best Individual':[
                f'{len(bestIndividual.nodeGenes)-len(nodeGenes)} hidden nodes',
                f'{len(bestIndividual.connectionGenes)} connection genes',
                f'{sorted(fitnesses, key=lambda x: x[1])[-1 ][1]} Fitness'
            ],
            'Species': f'{amtOfSpecies} Total Species',
            'Average Fitness':sum([i[1] for i in fitnesses])/len(fitnesses)
        })
        if generation % 10 == 0 and generation != 0:
            with open('bestNetwork.pkl', 'wb') as f:
                pkl.dump(bestIndividual, f)
else:
    with open('bestNetworkSent.pkl', 'rb') as f:
        bestIndividual = pkl.load(f)
    # bestIndividual = Network(nodeGenes=nodeGenes)
    
    # for i in range(25):
    #     bestIndividual.mutateAddConnection()
    #     bestIndividual.mutateEnableDisable(1.0, 0)
    #     if i % 5 == 0:
    #         bestIndividual.mutateAddNode()
    #     bestIndividual.mutateEnableDisable(1.0, 0)

    # print('start')
    # fig, axs = plt.subplots(1, 1, figsize=(14, 8))
    # fin=[]
    # for c in range(len(bestIndividual.connectionGenes)):
    #     if bestIndividual.connectionGenes[c].inLabel == 3:
    #         fin.append(bestIndividual.connectionGenes[c])
    # # bestIndividual.connectionGenes = fin
    # drawGenome(bestIndividual, ax=axs)
    # # drawGenome(networks[1], ax=axs[0][1])
    # # drawGenome(crossoverTwoNetworks(networks[0], networks[1], False), ax=axs[1][0])
    # plt.tight_layout()
    # # plt.show()
    # Game(bestIndividual).gameStep()
    # # quit()


# for i in range(len(networks)):
#     fig, axs = plt.subplots(1, 1, figsize=(14, 8))
#     drawGenome(networks[i], ax=axs)
#     # drawGenome(networks[1], ax=axs[0][1])
#     # drawGenome(crossoverTwoNetworks(networks[0], networks[1], False), ax=axs[1][0])
#     plt.tight_layout()
#     plt.show()



if True: 
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    drawGenome(bestIndividual, ax=axs[0][0])
    # drawGenome(networks[1], ax=axs[0][1])
    # drawGenome(crossoverTwoNetworks(networks[0], networks[1], False), ax=axs[1][0])
    plt.tight_layout()
    plt.show()

    game = Game(bestIndividual)

    while game.gameOngoing:
        game.gameStep()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.gameOngoing = False

        screen.fill((50, 50, 50))
        for segment in game.snake.body:
            if segment == game.snake.body[0]:  
                segmentColor = (180, 180, 180)
            else:
                segmentColor = (150, 150, 150)
            pygame.draw.rect(screen, segmentColor, (segment[0], segment[1], game.snake.stepSize, game.snake.stepSize)) # x, y, width, height
        pygame.draw.rect(screen, (255, 255, 255), (game.apple.x, game.apple.y, game.apple.stepSize, game.apple.stepSize)) # x, y, width, height

    

        pygame.display.flip()
        pygame.time.delay(100)
    print(game.fitness())
    pygame.quit()