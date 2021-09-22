import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

"""
Função: perceptron
Desenvolvida em: Python 3.9 em 14/09/2021
Bibliotecas Utilizadas: numpy 1.21.2 e matplotlib 3.4.3 
Finalidade: Ajustar os pesos repetidamente do neurônio para minimizar alguma medida de erro no conjunto de treinamento
Parametros: W:          Matriz de pesos; para cada neurônio k = 1
            b:          Vetor (k x 1) com valor do bias de cada neurônio
            X:          Matriz (m x N) com as amostras dos dados em colunas
            yd:         Matriz (k x N) com as saídas desejadas para cada amostra de dados(X)
            alfa:       Taxa de correção do peso; taxa de aprendizagem
            max_epocas: Valor máximo de épocas de treinamento
            tol:        Erro máximo tolerável
Retorno: W: Matriz de pesos ajustada
         b: bias ajustado
         VetorSEQ: Somatório dos erros quadráticos por épocas     
"""
def perceptron(W, b, X, yd, alfa, max_epocas, tol):
    N = len(X)  # Número de amostras de X
    seq = tol   # Somatório dos erros quadráticos
    epoca = 1   # Responsável por contar o número de epocas em que o neurônio foi treinado
    vetorSEQ = np.array([])  # Inicia o vetor do somatorio dos erros quadráticos
    while (epoca <= max_epocas) and (seq >= tol):
        seq = 0
        for i in range(N):   # percorre as amostras de X
            y = yperceptron(W, b, np.array([X[i]]).T)  # Determina a saída do neurônio em treinamento
            print(f'Saída da função yeperceptro(W={W}, b={b}, X={np.array([X[i]]).T}) = {y}')
            erro = yd[i] - y              # Calcula o erro
            W = W + (alfa * erro * X[i])  # Atualiza o vetor de pesos
            b = b + (alfa * erro)         # Atualiza o valor do bias
            seq = seq + (erro ** 2)
        vetorSEQ = np.append(vetorSEQ, seq)  # Incrementa o vetor do somatorio dos erros quadráticos
        epoca += 1                           # Incrementa a época de treinamento
    return W, b, vetorSEQ
### Inicia a definição dos dados para a função lógica AND

W = np.random.uniform(low=-1, high=1, size=(2))  # Determina aleatoriamente os pesos
W_entrada = W
b = np.random.uniform(low=-1, high=1, size=(1))  # Determina aleatoriamente o bias
b_entrada = b
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])   # Amostra dos dados
yd = [0, 0, 0, 1]                                # Saídas desejadas
alfa = 1.2                                       # Taxa de aprendizagem
max_epocas = 10                                  # Número máximo de épocas de treinamento
tolerancia = 0.001                               # Erro máximo tolerável
W, b, VetorSEQ = perceptron(W, b, X, yd, alfa, max_epocas, tolerancia)
print('-----------------------------------------------------------')
print('Executando para a função lógica AND sendo yd = [0, 0, 0, 1]')
print(f'W de entrada = {W_entrada}')
print(f'b de entrada = {b_entrada}')
print(f'W de saída = {W}')
print(f'b de saída = {b}')
print(f'VetorSEQ = {VetorSEQ}')
print(f'Épocas = {len(VetorSEQ)-1}')

plt1 = plt
plt1.title('Função Lógica AND')
plt1.xlabel('Épocas')
plt1.ylabel('SEQ')

plt1.plot(list(range(0, len(VetorSEQ))), VetorSEQ.tolist(), linewidth=2)  # plota reta de separação
plt1.axis([0, 10, 0, 5])
plt1.grid(True)
plt1.show()

### Inicia a definição dos dados para a função lógica OR
W = np.random.uniform(low=-1, high=1, size=(2))
W_entrada = W
b = np.random.uniform(low=-1, high=1, size=(1))
b_entrada = b
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
yd = [0, 1, 1, 1]
alfa = 1.2
max_epocas = 10
tolerancia = 0.001
W, b, VetorSEQ = perceptron(W, b, X, yd, alfa, max_epocas, tolerancia)
print('-----------------------------------------------------------')
print('Executando para a função lógica OR sendo yd = [0, 1, 1, 1]')
print(f'W de entrada = {W_entrada}')
print(f'b de entrada = {b_entrada}')
print(f'W de saída = {W}')
print(f'b de saída = {b}')
print(f'VetorSEQ = {VetorSEQ}')
print(f'Épocas = {len(VetorSEQ)-1}')
print('-----------------------------------------------------------')

plt2 = plt
plt2.title('Função Lógica OR')
plt2.xlabel('Épocas')
plt2.ylabel('SEQ')
plt2.plot(list(range(0, len(VetorSEQ))), VetorSEQ.tolist(), linewidth=2)
plt2.axis([0, 10, 0, 5])
plt2.grid(True)
plt2.xlabel('X Label')
plt2.ylabel('Y Label')
plt2.show()
