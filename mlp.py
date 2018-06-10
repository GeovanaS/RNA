######################################################################################################################################################## 
#  Trabalho FIA - Rede Neural Artificial                                                           				    			    				   #
#  Redes Neurais Multiplayer Perception capaz de classificar corretamente casos de cancer de mama  				    			    				   #
#  Grupo: Gabriella Selbach, Geovana Silveira, Luiza Cruz                                          				    			    				   #
########################################################################################################################################################

# importa classe MLPClassifier que implementa o algoritmo de multiplayer perceptron que realiza o treinamento usando backpropagation
from sklearn.neural_network import MLPClassifier
# importa e retorna o dataset do breast cancer wisconsin
from sklearn.datasets import load_breast_cancer
# importa modelo train_test_split para realizar a divisao do dataset em conjuntos de treino e teste
from sklearn.model_selection import train_test_split 
# importa metrica para o calculo do erro medio quadratico
from sklearn.metrics import mean_squared_error


# carrega o dataset
dados = load_breast_cancer()

# organiza os dados
classificacao = dados['target_names'] # armazena as classificacoes disponiveis para os casos de cancer de mama: malignant e benign
caracteristicas = dados['feature_names'] # armazena as caracteristicas dos tumores como por exemplo, o seu tamanho, raio e textura 
x = dados['data']  # armazena os valores referentes as caracteristicas dos tumores
y = dados['target'] # armazena o mapeamento para binario das classificacao dos casos representando tumores malignos como 0 e tumores benignos como 1 

# divide os dados para treino e teste utilizando uma porcentagem de 34% para teste e 66% para treinamento
x_treino,x_teste,y_treino,y_teste = train_test_split(x, y, test_size = 0.34)

# inicializa a classificacao passando uma tupla com numero de neuronios em cada camada e o numero maximo de iteracoes/epocas 
mlp = MLPClassifier(hidden_layer_sizes = (20,20,20), max_iter = 100)

# treina a classificacao
mlp.fit(x_treino, y_treino)




