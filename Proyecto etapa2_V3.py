import time
import random
import numpy as np
import pandas as pd
from multiprocessing import Process, Manager
import re
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

####    Algoritmo genetico
# Generación de números aleatorios para la carga
def cargas(sumaFinal,tamaño_ventana):
    numeros_random = []
    suma = 0
    while suma < sumaFinal:
        numeroA = random.randint(5, 15)
        if suma+numeroA > sumaFinal:
            numeros_random.append(sumaFinal-suma)
            break
        else:
            numeros_random.append(numeroA)
            suma = sum(numeros_random)
    if len(numeros_random) % tamaño_ventana != 0:
        resto = len(numeros_random) % tamaño_ventana
        if resto != 0:
            elementos_faltantes = tamaño_ventana - resto
            numeros_random.extend([0] * elementos_faltantes)
    return numeros_random
        
# generar la población

def generar_poblacion(num_individuos,tamaño_ventana):
    poblacion = []
    numeros = list(range(1,tamaño_ventana+1))
    for _ in range(num_individuos):
        random.shuffle(numeros)
        cadena = " ".join(map(str, numeros))
        poblacion.append(cadena)
    #print(poblacion)
    return poblacion
    
# fitness
def calcular_fitness(asignacion_procesos, maxspan):
    utilizacion_procesadores = []

    for procesos in asignacion_procesos.values():
        carga_total = sum(procesos)
        utilizacion = carga_total / maxspan
        utilizacion_procesadores.append(utilizacion)

    apu = sum(utilizacion_procesadores) / len(utilizacion_procesadores)  # Utilización promedio
    fitness = (1 / maxspan) * apu
    return fitness

# selección por ruleta
def seleccion_por_ruleta(poblacion, fitness_poblacion):
    total_fitness = sum(fitness_poblacion)
    probabilidad_seleccion = [fit / total_fitness for fit in fitness_poblacion]
    papa1 = random.choices(poblacion, weights=probabilidad_seleccion)[0]
    papa2 = random.choices(poblacion, weights=probabilidad_seleccion)[0]
    return papa1, papa2

# Función para realizar la mutación
def mutacion(individuo, tasa_mut):
    if random.random() < tasa_mut:
        individuo = individuo.split()
        idx1, idx2 = random.sample(range(len(individuo)), 2)
        individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]
        individuo = ' '.join(individuo)
    return individuo

### cruzamiento
def cruzamiento(papa1, papa2, tasa_cruz, tasa_mut):
    if random.random() < tasa_cruz:
        papa1 = papa1.split()
        papa2 = papa2.split()
        punto_de_cruce = random.randint(1, len(papa1) - 1)

        hijo1 = papa1[:punto_de_cruce] + [x for x in papa2 if x not in papa1[:punto_de_cruce]]
        hijo2 = papa2[:punto_de_cruce] + [x for x in papa1 if x not in papa2[:punto_de_cruce]]
        hijo1 = ' '.join(hijo1)
        hijo2 = ' '.join(hijo2)
    else:
        hijo1 = papa1
        hijo2 = papa2
    return mutacion(hijo1,tasa_mut), mutacion(hijo2,tasa_mut)

### conversión de individuo a asignación de procesos
def conversion(individuo,carga,tamaño_ventana):
    asignacion_procesos = {i + 1: [] for i in range(num_procesadores)}
    indices = list(map(int, individuo.split()))
    
    x = 0
    while x < len(carga):
        for i, indice in enumerate(indices):
            clave = (i % num_procesadores)+1  # Calcular la clave en función del índice
            asignacion_procesos[clave].append(carga[(indice - 1)+x])  # Restar 1 para ajustar el índice
        x = x+tamaño_ventana
    maxspan = max(sum(procesos) for procesos in asignacion_procesos.values())
    fitness = calcular_fitness(asignacion_procesos, maxspan)
    return asignacion_procesos,fitness

def AG(num_twits,num_procesadores):
    tamaño_ventana = num_procesadores*2
    num_individuos = 10

    num_generaciones = 10 
    tasa_mut = 0.1  
    tasa_cruz = 0.8 

    sum_pros = []
    sum_pros.extend([0] * num_procesadores)
    # print(f"\n\t\tNúmero de procesadores {num_procesadores}")
    tamaño_ventana = num_procesadores*2
    carga = cargas(num_twits,tamaño_ventana)
    poblacion = generar_poblacion(num_individuos, tamaño_ventana)
    for x in range(num_generaciones):
        # print(f"\nGeneración {x + 1}")
        nueva_poblacion=[]
        fitness = [conversion(individuo,carga,tamaño_ventana)[1] for individuo in poblacion]
        
        mejor_fitness = max(fitness)
        mejor_individuo = poblacion[fitness.index(mejor_fitness)]
        mejor_asignacion, _ = conversion(mejor_individuo,carga,tamaño_ventana)
        
        # print("Mejor individuo: ", mejor_individuo)
        # print("Fitness del Mejor individuo: ", mejor_fitness)
        # print("Asignación de twitts por procesador del mejor individuo:")
        for procesador, procesos in mejor_asignacion.items():
            #print(f'Procesador {procesador}: {sum(procesos)}')
            sum_pros[procesador-1] = sum(procesos)
            
        for _ in range(num_individuos // 2):
            papa1, papa2 = seleccion_por_ruleta(poblacion, fitness)
            hijo1, hijo2 = cruzamiento(papa1, papa2, tasa_cruz, tasa_mut)
            nueva_poblacion.extend([hijo1, hijo2])
        poblacion = nueva_poblacion

    return sum_pros


#####
def analizar_texto(df, procesador, inicio):
    start_time = time.time()
    #print(df.head(3))


    df.drop_duplicates(subset='text', inplace=True)
    df.loc[inicio,'text']

    def cleaner(text):
        text = re.sub('<[^<]*>', '', text)
        emoticons = ''.join(re.findall(r'[:;=]-+[\)\(pPD]+', text))
        text = re.sub(r'\W+', ' ', text.lower()) + emoticons.replace('-', '')
        return text

    cleaner(df.loc[inicio,'text'])

    df.loc[:,'text'] = df['text'].apply(cleaner)

    # print (df.shape)
    # print(df.head())

    def remove_contractions(text):
        contraction_patterns = re.compile(r"(can't|won't|I'll|I've|we'll|who's|what's|where's|when's|it's|that's|there's|how's)\s")
        text = re.sub(contraction_patterns, '', text)
        return text

    def clean_tweets(text):
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        text = text.lower()  
        return text

    def remove_duplicates(text):
        words = text.split()
        unique_words = list(set(words))
        return ' '.join(unique_words)

    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def preprocess_tweet(tweet):
        tweet = remove_contractions(tweet)
        tweet = clean_tweets(tweet)
        tweet = remove_duplicates(tweet)
        tweet = remove_stopwords(tweet)
        tweet = lemmatize_text(tweet)
        return tweet

    # Aplicar la función de preprocesamiento a la columna 'text' del DataFrame
    df.loc[:,'preprocessed_text'] = df['text'].apply(preprocess_tweet)

    # Tokenizar el texto preprocesado utilizando TextBlob
    df.loc[:,'word_tokenization'] = df['preprocessed_text'].apply(lambda x: TextBlob(x).words)


    # ## Separamos los valores de la variable 'sentiment' en negativos y positivos para despues concatenarlos en una tabla

    df_positive = df[df['class']=='suicide']
    df_negative = df[df['class']=='non-suicide']
    df_review_imb = pd.concat([df_positive, df_negative])
    # print(df_negative.shape)
    # print(df_negative.head())
    rus = RandomUnderSampler(random_state=0)
    df_review_bal, df_review_bal['class']=rus.fit_resample(df_review_imb[['text']],df_review_imb['class'])

    # ## Se separa los datos con el 80% de los datos para entrenamiento y el 20% para testing

    train, test = train_test_split(df_review_bal, test_size=0.2, random_state=0)

    train_x, train_y = train['text'], train['class']
    test_x, test_y = test['text'], test['class']


    # ## Modelo Word2vec
    sentences = [text.split() for text in df['preprocessed_text']]
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

    def text_to_vector(text):
        words = text.split()
        vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        
        if not vectors:  
            return np.zeros(word2vec_model.vector_size)  
        
        vector_lengths = [len(v) for v in vectors]
        max_length = max(vector_lengths)
        vectors = [np.pad(v, (0, max_length - len(v))) for v in vectors]
        
        return np.mean(vectors, axis=0)

    train_word2vec_vectors = np.array([text_to_vector(text) for text in train_x], dtype='float32')
    test_word2vec_vectors = np.array([text_to_vector(text) for text in test_x], dtype='float32')

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Procesador {procesador}: Tiempo transcurrido: {elapsed_time:.2f} segundos\n")

def main(texto_a_analizaranalizar,num_procesadores):

    # Obtener el número total de filas
    total_filas = len(texto_a_analizar)
    print(f"El DataFrame tiene {total_filas} filas.")

    #Alg Gneteico
    tm_particion = AG(total_filas, num_procesadores)
    #print(tm_particion)

    # Inicializar un vector para almacenar las particiones
    particiones = []

    # Crear y almacenar cada partición en el vector
    if sum(tm_particion) != total_filas:
        print("La suma de los tamaños de partición no es igual al número total de filas.")
    else:
    # Inicializar un vector para almacenar las particiones
        particiones = []
    # Crear y almacenar cada partición en el vector
        for i in range(len(tm_particion)):
            inicio = sum(tm_particion[:i])
            #print(inicio)
            fin = sum(tm_particion[:i+1])
            #print(fin)
            particion = texto_a_analizar.iloc[inicio:fin]
            particiones.append(particion)

            inicio = []
            anterior = 0
    # Puedes imprimir la longitud de cada partición si lo deseas
        for i, particion in enumerate(particiones,0):
            #print(f"Partición {i} tiene {len(particion)} filas.")
            #print(particiones[i-1].head(10))
            inicio.append(anterior + tm_particion[i]-1)
            anterior = inicio[i]+1

        #print(inicio)

        procesadores = []
        for i in range(num_procesadores):
            procesadores.append(i+1)
        # Crear un proceso para cada procesador
        procesos = []
        for procesador in procesadores:
            proceso = Process(target=analizar_texto, args=(particiones[procesador-1], procesador,inicio[procesador-1]))
            procesos.append(proceso)

        # Iniciar los procesos
        for proceso in procesos:
            proceso.start()
    
        # Esperar a que todos los procesos terminen
        for proceso in procesos:
            proceso.join()

if __name__ == "__main__":
    # Leer el archivo CSV
    texto_a_analizar = pd.read_csv('Suicide_Detection.csv').head(100000)
    #texto_a_analizar = pd.read_csv('300tw.csv')
    for i in range(1,9):
        num_procesadores = i
        print('Numero de procesadores = ',num_procesadores)
        main(texto_a_analizar,num_procesadores)

    num_twitsA = [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,200000]
    for i in range(0,11):
        num_procesadores = 1
        print('Numero de tweets = ',num_twitsA[i])
        print('Numero de procesadores = ',num_procesadores)
        texto_a_analizar = pd.read_csv('Suicide_Detection.csv').head(num_twitsA[i])
        main(texto_a_analizar,num_procesadores)

        num_procesadores = 8
        print('Numero de procesadores = ',num_procesadores)
        main(texto_a_analizar,num_procesadores)


