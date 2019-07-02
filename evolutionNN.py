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

geneLR_range = [0.003,0.01]

def evolve():
    population = []
    
    for i in range(TOTAL_POPULATION):
        chromosome = []
        for g in range(TOTAL_GENE):
            chromosome.append(round(np.random.uniform(geneLR_range[0],geneLR_range[1]),4))
        population.append(chromosome)
    print(population)


if __name__ == '__main__':
    evolve()