import random

# Parametros
NUM_CROMOSOMAS = 50
NUM_GENES = 20
NUM_GENERACIONES = 200
PROB_CRUCE = 0.8
PROB_MUTACION = 0.05
PORCENTAJE_ELITISMO = 0.1

# Métodos
def crear_cromosoma():
    return [random.randint(0, 1) for _ in range(NUM_GENES)]

def evaluar_fitness(cromosoma:list):
    return sum(cromosoma)

def seleccion(cromosomas):
    return random.choice(cromosomas)

def cruzar(padre1:list, padre2:list):
    punto_cruza = random.randint(1, NUM_GENES - 1)
    hijo1 = padre1[:punto_cruza] + padre2[punto_cruza:]
    hijo2 = padre2[:punto_cruza] + padre1[punto_cruza:]
    return hijo1, hijo2

def mutar(cromosoma:list):
    for i in range(NUM_GENES):
        if(random.random() <= PROB_MUTACION):
            cromosoma[i] = 1 - cromosoma[i] #Cambiamos el valor de 1 a 0 y viceversa

# Implementación
poblacion = [crear_cromosoma() for _ in range(NUM_CROMOSOMAS)]

for generacion in range(NUM_GENERACIONES):
    nueva_poblacion = []

    num_elitismo = int(NUM_CROMOSOMAS * PORCENTAJE_ELITISMO)
    nueva_poblacion.extend(sorted(poblacion, key=evaluar_fitness, reverse=True)[:num_elitismo])

    for _ in range(NUM_CROMOSOMAS // 2):
        if random.random() <= PROB_CRUCE: 
            padre1 = seleccion(poblacion)
            padre2 = seleccion(poblacion)
            
            hijo1, hijo2 = cruzar(padre1, padre2)

            mutar(hijo1)
            mutar(hijo2)

            nueva_poblacion.extend([hijo1, hijo2])

    poblacion = nueva_poblacion

mejor_cromosoma = max(poblacion, key=evaluar_fitness)
print("Cromosoma óptimo: ", mejor_cromosoma)
print("Fitness: ", evaluar_fitness(mejor_cromosoma))