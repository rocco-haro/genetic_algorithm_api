# Rocco Haro & brandon sayers
# the supermodel class abstracts renewable models for one particular town

import models.stackedLSTM as modelBuilder_LSTM
import pandas as pd
import random
import models.workingNN as NN
import numpy as np
import csv
import math
import matplotlib.pyplot as plt

class GeneticAlg:
    def __init__(self, popSize, modelName):
        """ Genetic algorithm that finds optimal parameters.
            Input the size of a population
        """
        self.pop = [] # the population
        self.fitnessVals = [0 for x in range(len(self.pop))] # matched indexed with population

        self.modelName = modelName
        self.dataFrame = self.loadData()
        self.model = None
        self.popSize = popSize
        self.initPop()

    def loadData(self):
        # Pull all data from CSV file and
        # push into a dataframe for portability.
        df = pd.read_csv(self.modelName, index_col=0, skiprows=[1])
        df.index = pd.to_datetime(df.index)
        return df

    def randNum(self, typeR, _min, _max):
        # https://stackoverflow.com/questions/33359740/random-number-between-0-and-1-in-python
        x = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        if (typeR == "0"):
            return x
        else:
            return random.randint(_min, _max) * x

    def getNextGen(self):
        def pickPerson(self):
            totalLossInPi = math.fsum(self.fitnessVals)
            maxNum = math.ceil(totalLossInPi)
            personAt = self.randNum(1, 0, maxNum)
            currFit = 0
            for i in range(self.fitnessVals):
                if currFit > personAt:
                    return i
                currFit+= self.fitnessVals[0]

        def mate(self, p1, p2):
            holder = []
            childParams = dict()
            for key_p1, key_p2 in enumerate(p1, p2):
                whichPerson = self.randNum(0, "","") # return a number between 0 and 1
                if (whichPerson < 0.5):
                    childParams[key_p1] = p1[key_p1]
                else:
                    childParams[key_p2] = p2[key_p2]
            return childParams
        ######################################################
        # Begin:
        children = []
        person1 = -1
        person2 = -1
        for i in range(self.popSize):
            while(person1 == person2):
                # TODO based off of their fitness values, not just randomly
                person1 = pickPerson()
                person2 = pickPerson()
                x=0
            child = self.mate(person1, person2)
            children.append(child)
        return children

    def runToGeneration(self, maxGens):
        def runFitness():
            def getLoss(idx):
                """ return loss
                """
                self.model = modelBuilder_LSTM.StackedLSTM(dataFrame=self.dataFrame[self.modelName], modelName=(column + "/" + column + self.testNum))
                params = self.pop[idx]
                self.model.networkParams(params['ID'], n_input=params['n_input'],n_steps=params['n_steps'], n_hidden=params['n_hidden'], n_outputs=params['n_outputs'], n_layers=params['n_layers']   ) # should look like: networkParams(n_steps = params['n_steps'], n_layers = params['n_layers'] , ....)
                popOutAt = 0.5
                loss = self.model.trainKickOut(popOutAt) # Neeed to be implemented
                self.model = None
                return loss
            ## getLost ^

            allLosses = []
            for i in range(len(self.pop)):
                loss = getLoss(i)
                allLosses.append(loss)

            maxVal = np.amax(allLosses)
            return [(1- (x/maxVal)) for x in allLosses] # lower loss, greater fitness
        ## runFitness ^

        newPopulation = None
        for gen in range(maxGens):
            self.fitnessVals = runFitness()
            self.pop = self.getNextGen()
            return 0

    def createIndividual(self):
        # return an individual with a random set of network params
        # LSTM network params (self,ID, n_input = 1,n_steps = 11, n_hidden= 2, n_outputs = 5 , n_layers = 2, loading=False):
        params = dict()
        n = "" # nothing but a place holder of "nothing"
        params['id'] = -1 # denotes that its an LSTM used for genetic algorithm
        params['n_input'] = 1
        params['n_steps'] = self.randNum(n, 5, 100) # typeR, min, max
        params['n_hidden']= self.randNum(n, 1, 15)
        params['n_outputs'] = self.randNum(n, 1, 15)
        params['n_layers'] = self.randNum(n, 1, 10)

        return params

    def initPop(self):
        try:
            for x in range(self.popSize):
                indiv = createIndividual()
                self.pop.append(indiv)
        except:
            print("GeneticAlg says: Failed to initialize the population")
