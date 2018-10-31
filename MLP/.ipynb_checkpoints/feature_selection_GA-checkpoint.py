import numpy as np
from random import randint,randrange,sample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
    
def cal_pop_fitness(population, X_train, y_train, X_test, y_test, random_itens):
    # calcula o fit (no caso r2) para cada solucao
    fitness = []
    new_pop = []
    count = 1
    for pop in population:
        #print("Running {} models".format(count))
        #pop = sample(pop, random_itens)
        new_pop.append(pop)
        # Criando classificador 
        model = None
        model = RandomForestClassifier(n_estimators = 7, random_state = 42)
        model.fit(X_train[pop], np.ravel(y_train))
        results = model.predict(X_test[pop])
        fitness.append(roc_auc_score(y_test, results))
        count+=1
    return fitness,new_pop


def select_mating_pool(population, fitness, num_parents):
    # seleciona o numero de pais para a proxima iteracao
    # leva em conta as populacoes geradas, a lista de fitnes e o numero de pais 
    # cria um array vazio para ser o array de pais
    parents = []
    for num_parents in range(num_parents):
        # seleciona o id com fit maximo
        max_fitness_idx =  np.where(fitness == np.max(fitness))
        # id vem em formato de lista [[id]]
        max_fitness_idx = max_fitness_idx[0][0]
        # atribui a populacao de maior fit a um pai
        parents.append(population[max_fitness_idx])
        # atribui o id desse fit no array para -inf para nao ser solucionada de novo
        fitness[max_fitness_idx] = np.NINF
    return parents


def crossover(parents, offspring_size):
    # com os pais selecionados passamos para a etapa de crossover
    # o ponto de crossover entre pais normalmente e no centro
    crossover_point = int(offspring_size/2)
    offspring = []
    for k in range(offspring_size):
        this_offspring = []
        # para cada geracao
        # seleciono dois pais consecutivos
        parent1_idx = k % len(parents)
        parent2_idx = (k+1) % len(parents)
        # a geracao recebe o comeco de um pai
        this_offspring = parents[parent1_idx][:crossover_point]
        # e o final de outro
        this_offspring = this_offspring + parents[parent2_idx][crossover_point:]
        offspring.append(this_offspring)
    return offspring

        
def mutation(offspring_crossover, full_model_cols):
    # aplica uma mutacao randomica  em um gene unico de uma subpop
    final_offspring = []
    for population in offspring_crossover:
        in_pop = set(population)
        # available cols to randonfy
        available_random = list(set(full_model_cols) - set(in_pop))    
        # remove random elem from pop 
        to_remove = population[randrange(len(population))]
        population.remove(to_remove)
        # generate random value in len of original cols - already in population
        random_elem = randint(0, len(available_random)-1)
        to_add = available_random[random_elem]
        population.append(to_add)
        final_offspring.append(population)
        
    return offspring_crossover

def sanity_pop(actual_pop, must_pop):
    # garante que todos os itens da populacao original estejam na populacao final
    # garante tambem que a populacao final nao tenha duplicatas
    
    #remove duplicados
    actual_pop = list(set(actual_pop))
    
    # pega diferenca entre original e atual
    diff_elems = list(set(must_pop) - set(actual_pop))
    
    final_pop = actual_pop + diff_elems
    # garante que trend seja a ultima coluna
    try:
        final_pop.remove('trend')
        final_pop.append('trend')
    except:
        pass
    return final_pop
    
def optimize_ga(X,y, X_test, y_test, pop_size, cols_per_pop, num_generations, num_parents_mating, verbose=False):
    # numero de solucoes esperadas por populacao
    solution_per_pop = pop_size
    # tamanho da minha populacao
    num_cols = len(X.columns.values)
    # colunas originais do modelo
    model_cols = list(X.columns.values)
    new_population = []
    for i in range(solution_per_pop):
        t = sample(model_cols, cols_per_pop)
        new_population.append(t)
    
    if verbose:
        print("Optimizing with {} generations".format(num_generations))
    for id_gen, generations in enumerate(range(num_generations)):
        if verbose:
            print("Generation {}".format(id_gen))
        # mede o fit de cada popuilacao
        #print("\tCalc fit...")
        fitness,new_population = cal_pop_fitness(new_population, X, y,X_test, y_test, pop_size)
        # seleciona os melhores pais na populacao para crossover
        #print("\tParent mating...")
        parents = select_mating_pool(new_population, fitness, 
                                           num_parents_mating)

        # gera a proxima populacao utilizando crossover
        #print("\tCross overing...")
        offspring_crossover = crossover(parents, pop_size)
        # adiciona mutacao para a offspring
        #print("\tMutating...")
        offspring_mutation = mutation(offspring_crossover, model_cols)
        # cria nova pipulacao baseado nos pais e na offpsring
        new_population = []
        new_population[0:len(parents[0])] = parents
        new_population[len(parents[0]):] = offspring_mutation
        #print("\tFinal Fitting")
        fitness,_ =  cal_pop_fitness(new_population, X, y,X_test, y_test, pop_size)
        # melhor resultado na iteracao
        if verbose:
            print("Best result : {}".format(np.max(fitness)))
        
    # apos iterar por todas as geracoes pegar a melhor solucao
    # calculamos o fit para cada solucao
    final_fitness,_ =  cal_pop_fitness(new_population, X, y,X_test, y_test, pop_size)

    # pegamos o elemento de melhor fit
    best_match_idx = np.where(final_fitness == np.max(final_fitness))[0][0]
    final_pop  = new_population[best_match_idx]
    return final_pop
    