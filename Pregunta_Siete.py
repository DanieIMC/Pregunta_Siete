import random
import numpy as np

# Definir la función de fitness común para ambos incisos
def fitness_function(x):
    try:
        result = (x ** (2 * x)) - 1
        if np.iscomplex(result):  # Si el resultado es complejo, penalizarlo
            return -9999  # Penalización severa para evitar problemas
        return result.real  # Solo la parte real
    except ValueError:  # Manejo de casos que pueden producir errores
        return -9999

############################
# INCISO A - CON USO DE DEAP
############################

from deap import base, creator, tools, algorithms

# Configurar la estructura del algoritmo genético con DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizar la función
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Definir los límites para x entre -2 y 2
toolbox.register("attr_float", random.uniform, -2, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", lambda ind: (fitness_function(ind[0]),))
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Cruce
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Mutación
toolbox.register("select", tools.selTournament, tournsize=3)  # Selección

# Algoritmo genético en DEAP
def run_with_deap():
    random.seed(42)
    pop = toolbox.population(n=10)  # Población inicial
    ngen = 3  # Número de generaciones
    cxpb = 0.5  # Probabilidad de cruce
    mutpb = 0.2  # Probabilidad de mutación

    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Ejecutar el algoritmo
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats=stats, verbose=True)

    # Mostrar la mejor solución
    best_individual = tools.selBest(pop, 1)[0]
    print(f"Mejor individuo con DEAP: {best_individual}, Fitness: {best_individual.fitness.values[0]}")

#############################
# INCISO B - SIN USO DE DEAP
#############################

# Inicializar la población
def init_population(size, min_value, max_value):
    return [random.uniform(min_value, max_value) for _ in range(size)]

# Selección por torneo
def tournament_selection(population, fitnesses, k=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), k)
        selected.append(max(tournament, key=lambda t: t[1])[0])
    return selected

# Cruce (promedio de dos padres)
def crossover(parent1, parent2):
    return (parent1 + parent2) / 2

# Mutación (perturbar ligeramente)
def mutate(individual, mutation_rate=0.1):
    return individual + random.uniform(-mutation_rate, mutation_rate)

# Algoritmo genético sin DEAP
def genetic_algorithm(pop_size=10, generations=3, min_value=-2, max_value=2, crossover_prob=0.5, mutation_prob=0.2):
    population = init_population(pop_size, min_value, max_value)

    for gen in range(generations):
        # Evaluar fitness
        fitnesses = [fitness_function(ind) for ind in population]
        
        # Selección
        selected_population = tournament_selection(population, fitnesses)
        
        # Cruce
        next_generation = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1] if i+1 < len(selected_population) else selected_population[0]
            if random.random() < crossover_prob:
                offspring1 = crossover(parent1, parent2)
                offspring2 = crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1, parent2
            next_generation.extend([offspring1, offspring2])
        
        # Mutación
        next_generation = [mutate(ind) if random.random() < mutation_prob else ind for ind in next_generation]
        
        # Actualizar la población
        population = next_generation[:pop_size]

        # Mostrar la mejor solución de la generación actual
        fitnesses = [fitness_function(ind) for ind in population]
        best_individual = population[np.argmax(fitnesses)]
        print(f"Generación {gen+1} - Mejor individuo sin DEAP: {best_individual}, Fitness: {max(fitnesses)}")
    
    # Devolver la mejor solución
    return best_individual

if __name__ == "__main__":
    print("Ejecutando el algoritmo genético con DEAP:")
    run_with_deap()

    print("\nEjecutando el algoritmo genético sin DEAP:")
    best = genetic_algorithm()
    print(f"Mejor solución sin DEAP: {best}")
