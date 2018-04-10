import SleepingThread.ga as ga
import random

random.seed(5)

def qual_fun(individ_data):
    return individ_data[1]+0.1*(individ_data[0]+individ_data[2])

root = ga.Individual(data=[0,0.0,0],limitations=[[0,10],[0.0,50.0],[-1,20]])
population = ga.Population(root,qual_fun,save_prev_population=True)

population.generateRandom()
for i in xrange(200):
    population.makeNextGeneration()
    print "Best qual: ",population.selectBest(population.best)[0].quality,\
            " ext: ",population.calculateExtinctionValue()

