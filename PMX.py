import random

def Own_PMX(P1, P2):
    L = len(P1)
    H1 = [None] * L
    H2 = [None] * L

    l1 = random.randint(0, L - 1)
    l2 = random.randint(l1 + 1, L)

    # Llenamos los hijos con los valores de los intervalos definidos del padre contrario
    H1[l1:l2] = P2[l1:l2]
    H2[l1:l2] = P1[l1:l2]
    # print(f"{H1}\n{H2}\n")

    for h in range(2):
        if h == 0:
            Ha = H1
            Pa = P1
            Pc = P2
        else:
            Ha = H2
            Pa = P2
            Pc = P1

        # Completamos con los valores del mismo padre que no se encuentren en el hijo
        for i in range(len(Ha)):
            if (Ha[i] == None):
                if (Pa[i] not in Ha):
                    Ha[i] = Pa[i]
        
        # Terminamos completando al hijo con los valores del padre contrario que no se encuentren
        #  en el orden de izquierda a derecha
        for i in range(len(Pc)):
            if (Pc[i] not in Ha):
                Ha[Ha.index(None)] = Pc[i]
        
        if h == 0:
            H1 = Ha
        else:
            H2 = Ha

    # print(f"{H1}\n{H2}")
    # print(len(set(H1)) == len(H1) and len(set(H2)) == len(H2)) # Comprobar que no hay elementos repetidos en ambas listas, todo se hizo correctamente
    
    return H1, H2

# https://youtu.be/hnxn6DtLYcY?si=uNVMSb1FYrl9qMYt&t=430