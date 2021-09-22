import numpy as np

"""
Função: yperceptron
Desenvolvida em: Python 3.9 em 06/09/2021
Bibliotecas Utilizadas: numpy 1.21.2  
Finalidade: Via função degrau (threshold) determina a saída do neurônio
Parametros: W: Vetor de pesos de cada entrada do neurônio
            b: Valor do bias 
            X: Matriz de amostra de dados         
Retorno: y: Saída no neurônio para cada amostra X
"""
def yperceptron(W, b, X):
    y = np.dot(W, X) + b
    for i in range(len(y)):
        if y[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y
