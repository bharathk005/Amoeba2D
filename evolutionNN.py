# START
# Generate the initial population
# Compute fitness
# REPEAT
#     Selection
#     Crossover
#     Mutation
#     Compute fitness
# UNTIL population has converged
# STOP
import controlNN as cnn
import numpy as np

TOTAL_POPULATION = 10
TOTAL_GENE = 1
SELECTION = 5
MUTATION_CHANCE = 0.1
#select 5 and crossover only the top 4

geneLR_range = [0.003,0.01]

population = []
fitness = []

def select():
    temp = []
    global population
    for _ in range(SELECTION):
        f = np.argmax(fitness)
        temp.append(population.pop(f))
        fitness.pop(f)
    population.clear()
    fitness.clear()
    population = temp
    print("Selection")
    print(population)

def crossover():
#select 5 and crossover only the top 4 to produce 3.
#3 cross + 1 retained + 4 parents + 2 random
    for i in range(SELECTION - 2):
        chromosome = []
        parent1 = population[i]
        parent2 = population[i+1]
        cross = round((parent1[0] + parent2[0])/2 , 4)
        #add additional genes to this chromosome
        chromosome.append(cross)
        population.append(chromosome)
    #append 2 random genes at the end. 
    for i in range(TOTAL_POPULATION- len(population)):
        chromosome = []
        for g in range(TOTAL_GENE):
            chromosome.append(round(np.random.uniform(geneLR_range[0],geneLR_range[1]),4))
        population.append(chromosome)
    print("Crossover")
    print(population)

def mutation():
    for p in population:
        randi = np.random.uniform(0,1)
        if MUTATION_CHANCE > randi:
            print("++Mutated++",randi)
            delt = round(np.random.uniform(-0.005,0.005),4)
            print(delt)
            p[0] += delt

def simulate_reward():
    mean_reward = round(np.random.uniform(50,100),2)
    max_reward = round(np.random.uniform(75,300),2)
    return mean_reward,max_reward

def evolve():
    select()
    crossover()
    mutation()
    for p in population:
        mean_reward,max_reward = simulate_reward() #cnn.trainNN(p[0])
        fitness.append(mean_reward)

def populate():
    
    for i in range(TOTAL_POPULATION):
        chromosome = []
        #add loops for genes
        for g in range(TOTAL_GENE):
            chromosome.append(round(np.random.uniform(geneLR_range[0],geneLR_range[1]),4))
        population.append(chromosome)
    print("initial population")
    print(population)
    for p in population:
        mean_reward,max_reward = simulate_reward() #cnn.trainNN(p[0])
        fitness.append(mean_reward)
    print("initial fitness")
    print(fitness)



if __name__ == '__main__':
    populate()
    evolve()