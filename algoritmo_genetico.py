import random
from abc import ABC, abstractmethod

class AlgoritmoGenetico(ABC):
    def __init__(self, low_val = 0, top_val = 1, num_cromosomas = 50, num_genes = 20, num_generaciones = 200, prob_cruce = 0.8, prob_mutacion = 0.05, porcentaje_elitismo = 0.05):
        self.low_val = low_val
        self.top_val = top_val
        self.num_cromosomas = num_cromosomas
        self.num_genes = num_genes
        self.num_generaciones = num_generaciones
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.porcentaje_elitismo = porcentaje_elitismo

    def crear_cromosoma(self):
        return [random.randint(self.low_val, self.top_val) for _ in range(self.num_genes)]

    @abstractmethod
    def evaluar_fitness(self, cromosoma:list):
        pass

    def seleccion(self, cromosomas:list):
        return random.choice(cromosomas)

    def cruzar(self, padre1:list, padre2:list):
        punto_cruza = random.randint(1, self.num_genes - 1)
        hijo1 = padre1[:punto_cruza] + padre2[punto_cruza:]
        hijo2 = padre2[:punto_cruza] + padre1[punto_cruza:]
        return hijo1, hijo2

    def mutar(self, cromosoma:list):
        for i in range(self.num_genes):
            if(random.random() <= self.prob_mutacion):
                cromosoma[i] = random.randint(self.low_val, self.top_val)

    def best(self, poblacion:list):
        return max(poblacion, key=self.evaluar_fitness)

    def fit(self, poblacion:list = None):
        if poblacion == None:
            poblacion = [self.crear_cromosoma() for _ in range(self.num_cromosomas)]

        for _ in range(self.num_generaciones):
            nueva_poblacion = []

            num_elitismo = int(self.num_cromosomas * self.porcentaje_elitismo)
            nueva_poblacion.extend(sorted(poblacion, key=self.evaluar_fitness, reverse=True)[:num_elitismo])

            while (len(nueva_poblacion) + 2) <= (self.num_cromosomas):

                if random.random() <= self.prob_cruce: 
                    padre1 = self.seleccion(poblacion)
                    padre2 = self.seleccion(poblacion)
                    
                    hijo1, hijo2 = self.cruzar(padre1, padre2)

                    self.mutar(hijo1)
                    self.mutar(hijo2)

                    nueva_poblacion.extend([hijo1, hijo2])

            poblacion = nueva_poblacion

        return poblacion