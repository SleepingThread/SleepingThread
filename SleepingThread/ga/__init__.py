# -*- coding: utf-8 -*-
"""
This module implements Genetic Algorithms for selection best individual
"""

import copy
import random
import numpy as np

class Operations(object):
    @staticmethod
    def mutation(individ,prob=0.4,mutation_prob=0.6):
        """
        mutation_prob - probability of individ mutation
        prob - probability of gene mutation
        """
        data = individ.data
        limit = individ.limitations
        if random.random() < mutation_prob:
            for i in xrange(len(data)):
                # depending on type create new value
                # with respect to individ.limitations
                if random.random() < prob:
                    if type(data[i]) == int:
                        data[i] = random.randint(limit[i][0],limit[i][1])
                    elif type(data[i]) == float:
                        scale = limit[i][1]-limit[i][0]
                        data[i] = limit[i][0]+scale*random.random()

            individ.quality = None

        return individ

    @staticmethod
    def crossingover(individ1,individ2,prob = 0.4,cross_prob = 0.6):
        """
        prob - probability of crossingover for gene
        cross_prob - probability of crossingover for individ
        """
        data1 = individ1.data
        data2 = individ2.data
        if len(data1) != len(data2):
            raise Exception("individ genes has different length")
        if random.random() < cross_prob:
            for i in xrange(len(data1)):
                if random.random() < prob:
                    # make crossingover
                    _sw = data1[i]
                    data1[i] = data2[i]
                    data2[i] = _sw

            individ1.quality = None
            individ2.quality = None

        return individ1,individ2



class Individual(object):
    """
    Class supports list of floats or ints with limitations
    """
    def __init__(self,data,limitations=None):
        # store genes for individual
        self.data = copy.deepcopy(data)
        self.quality = None
        if limitations is None:
            self.limitations = len(self.data)*[None]
        else:
            self.limitations = copy.deepcopy(limitations)

        self.generation = 0

        return

    def setQuality(self,qual):
        self.quality = qual
        return

    def __str__(self):
        return "Individual: "+str(self.data)

    @staticmethod
    def addGeneration(individ):
        individ.generation += 1
        return


class Population(object):
    def __init__(self,root,quality_fun,size=100,best_size=20,init_population=None,random_seed=0,save_elite=True,save_prev_population=False):
        """
        root - first individual or list of individuals in all population
        quality_fun - function for calculating quality of one individual
        """
        if init_population is None:
            self.population = []
        else:
            self.population = copy.deepcopy(init_population)

        self.root = root
        self.limitations = root.limitations

        self.size = size
        self.save_elite = save_elite
        self.save_prev_population = save_prev_population

        self.quality_fun = quality_fun

        self.generation = 0

        self.best_size = best_size
        self.best = []

        return

    def generateRandom(self):
        for i in xrange(self.size):
            new_individ = copy.deepcopy(self.root)
            Operations.mutation(new_individ,prob=1.0,mutation_prob=1.0)
            self.population.append(new_individ)
        return

    def makeNewPopulation(self):
        """
        take self.population and make new
        """
        new_population = list(map(copy.deepcopy,self.population))
        map(Individual.addGeneration,new_population)

        # make mutation and crossingover
        map(lambda el: Operations.mutation(el,prob=0.4,mutation_prob=0.6),new_population)

        for i in xrange(len(new_population)):
            for j in xrange(i+1,len(new_population)):
                Operations.crossingover(new_population[i],new_population[j],\
                        prob=0.4,cross_prob=0.6)

        return new_population

    def selectBestArgs(self,individs,selection_size=1):
        #extract qualities from individs
        individs_quals = [el.quality for el in individs]
        inds = np.argsort(individs_quals)
        return inds[-selection_size:]

    def selectBest(self,individs,selection_size=1):
        # extract qualities from individs
        individs = np.asarray(individs)
        individs_quals = [el.quality for el in individs]
        inds = np.argsort(individs_quals)
        return individs[inds[-selection_size:]].tolist()

    def makeSelection(self,tourn_size = 3):
        """
        take self.population and make <<natural selection>>
        """

        pop = self.population

        # selected population
        sel_population = []

        # tounament scheme

        selected = set()

        for i in xrange(self.size):
            # select tourn_size = 3 from population
            tourn_member = []
            tourn_inds = []
            while len(tourn_inds) < tourn_size:
                _val = random.randint(0,len(pop)-1)
                if _val in selected:
                    continue

                tourn_inds.append(_val)
                tourn_member.append(pop[tourn_inds[-1]])

            ind = self.selectBestArgs(tourn_member)[0]
            selected.add(tourn_inds[ind])
            sel_population.append(pop[tourn_inds[ind]])

        self.population = sel_population

        return sel_population

    def updateBest(self):
        individs = []
        individs.extend(self.best)
        individs.extend(self.population)
        self.best = self.selectBest(individs,self.best_size)
        return

    def calculateQuality(self,new_population):
        for el in new_population:
            el.quality = self.quality_fun(el.data)

        return

    def calculateExtinctionValue(self):
        arr = np.array([el.data for el in self.population])
        average = np.average(arr,axis=0)
        arr = (arr-average)**2
        arr = 1.0/len(self.population) * np.sum(arr,axis=0)
        val = 0.0
        for i in xrange(len(arr)):
            val += arr[i]/(self.limitations[i][1]-self.limitations[i][0])

        return val

    def makeNextGeneration(self):
        old_population = self.population
        new_population = self.makeNewPopulation()

        # calculate quality with quality_fun
        self.calculateQuality(new_population)

        # make new population
        if self.save_prev_population:
            self.population.extend(new_population)
        else:
            self.population = new_population

        # update self.best
        self.updateBest()

        # select from population
        self.makeSelection()

        self.generation += 1

        return
