import numpy as np
import random
import matplotlib.pyplot as plt

# variáveis de altura 0, largura 1 e comprimento2 do motor têm, cada uma, 20 valores com valor 0 (valor em falta)
# coluna 12 tem 20 outliers > 90

# ---------------------------------------- Carregamento dos dados ------------------------------------------------------

data = np.load("projeto_dataset.npy")
labels = np.load("names_dataset.npy")

lines = data.shape[0]
colunas = data.shape[1]

transmissao = data[:, 6]

# ---------------------------------- Calculo dos coeficientes de correlação --------------------------------------------
CC_list = []
for col in range(colunas):
    CC_list.append(np.abs(np.corrcoef(data[:, col], transmissao)[0][1]))

# ------------------------------------- Representação grafica das variaveis ---------------------------------------------
def graficRepresentation():
    for col in range(colunas):
        plt.figure(col)
        for row in range(lines):
            #print(f'{row}={data[:,col][row]}')
            plt.scatter(row,data[:,col][row])


# -------------------------- Verificando se o VOLUME do motor tem alguma correlação significativa ----------------------
def vol_motor():
    vol_motor = []
    for i in range(lines):
        vol = data[i, 0] * data[i, 1] * data[i, 2]
        vol_motor.append(vol)

    corre_motor = np.corrcoef(vol_motor, transmissao)[0][1]
    print(corre_motor)

    #O volume do motor não tem relação significativa com a transmissão

# --------------------------------------- Seleção de variaveis ---------------------------------------------------------
def k(val1, val2):
    lst = []
    for idx in range(len(CC_list)):
        if val1 < CC_list[idx] < val2:
            lst.append(idx)
    return lst


CCs = k(0.20, 0.9)

new_data = []
new_label = []

for col in CCs:
    new_data.append(data[:, col])  # Esse passo transpõe os dados
    new_label.append(labels[col])  # Linha vira coluna, coluna vira linha

new_data.append(transmissao)
# ---------------------------------------------------
# def filtro_std():

#---------------------------Removendo os outliers ja conhecidos em City_mpg--------------------------------------------
# Metodo KNN
def outliers_KNN(variavel):
    tratar = variavel
    k = 3  # numero de vizinhos

    for i in range(lines):
        vizinhos = []
        if tratar[i] > 90:  # Valor 90 escolhido pois o  professor falou que outliers eram > 90

            for j in range(1, k + 1):
                vizinhos.append(tratar[i - k])
                vizinhos.append(tratar[i + k])

            mean = np.mean(vizinhos)
            # print(city_mpg[i], mean, vizinhos)
            tratar[i] = mean

    variavel = tratar

outliers_KNN(new_data[1]) # city_mpg

new_data = np.array(new_data)
new_label = np.array(new_label)
# -------------------------------- Removendo outlier de highway_mph ----------------------------------------------------
# Metodo Filtro através do desvio padrão
def outliers_filter(data_filter, fator):
    mean = np.mean(data_filter)
    desvio = np.std(data_filter)

    limMax = mean + fator * desvio
    limMin = mean - fator * desvio


    outlierMax = np.where(data_filter >= limMax)[0]
    outlierMin = np.where(data_filter <= limMin)[0]

    data_filter[outlierMax] = limMax
    data_filter[outlierMin] = limMin

outliers_filter(new_data[2, :], 2) # hightway_mpg
# ----------------------------------------- Normalização dos dados -----------------------------------------------------
for i in range(new_data.shape[0] - 1):
    new_data[i, :] = new_data[i, :] - new_data[i, :].min()
    new_data[i, :] = new_data[i, :] / new_data[i, :].max()

new_lines = new_data.shape[0]  # Originalemtne colunas, são as variaveis (ainda segue a ordem de "Labels")
new_colunas = new_data.shape[1]  # Originalmente linhas, são os dados

# -------------------------------- Separando os dados em Treino/Validação ----------------------------------------------

random.shuffle(new_data[0])  # Shuffle apenas nas linhas
data_treino = np.array(new_data[:3, :3574])
resul_treino = np.array(new_data[3, :3574])

data_validacao = np.array(new_data[:3, 3574:])
resul_avaliacao = np.array(new_data[3, 3574:])

# --------------------------------------------- Criação dos modelos ----------------------------------------------------
# _____________________________________________ Funções auxiliares _____________________________________________________


def moda(lst):
    if np.sum(lst) >= len(lst)/2:
        return 1
    else:
        return 0

# ================================================== Método KNN ========================================================

def knn_point(P, data_knnP, resul):  # Função que classifica um ponto apenas
    NK = 3
    dist = np.zeros(data_knnP.shape[0])

    for i in range(0, data_knnP.shape[0]):
        di = np.linalg.norm(P - data_knnP[:, i])
        dist[i] = di

    idx = np.argsort(dist)
    res = resul[idx[0:NK]]
    classe = moda(res)

    return classe


def knn(data_knn, resul):  # Função para classificar todos os pontos
    classification = []
    for line in range(data_knn.shape[1]):
        classification.append(knn_point([data_knn[0, line], data_knn[1, line], data_knn[2, line]], data_knn, resul))

    classification = np.array(classification)

    # Plot dos pontos em 3 dimensões
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection ='3d')
    for lin in range(data_knn.shape[1]):
        xs = data_knn[0, lin]   # Torque
        ys = data_knn[1, lin]   # city_mpg
        zs = data_knn[2, lin]   # highway_mpg

        if classification[lin] == 0:
            p0 = plt.scatter(xs, ys, zs, 'b')
        else:
            p1 = plt.scatter(xs, ys, zs, 'r')



    plt.title("Representação grafica da classificação")
    fig.legend((p0, p1), ('Elétrico', 'Combustão'), loc='upper left')

    ax.set_xlabel("Torque")
    ax.set_ylabel("city_mpg")
    ax.set_zlabel("highway_mpg")

    plt.show()
    """
# ____________________________________________________________________________________

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for idx in range(0, len(classification)):
        if classification[idx] == resul[idx] and classification[idx] == 1:
            TP += 1
        if classification[idx] == resul[idx] and classification[idx] == 0:
            TN = TN + 1
        if classification[idx] == 1 and resul[idx] == 0:
            FN = FN + 1
        if classification[idx] == 0 and resul[idx] == 1:
            FP = FP + 1

    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    print("\n ------------------ KNN ------------------")
    print(" SE - sensibilidade  =", round(SE, 3))
    print(" SP - Especificidade =", round(SP, 3))

# ======================================= Fronteira de Decisão =========================================================


def fronteria_decisao():
    # Calculo dos Minimos quadrados
    UM = np.ones((data_treino.shape[1], 3574))
    X1 = np.concatenate((data_treino, UM), axis=0)
    id0 = np.where(transmissao == 0)[0]
    transmissao[id0] = -1

    LSQ = np.linalg.lstsq(X1, transmissao, rcond=None)
    PAR = LSQ[0]

    # Fronteira de decisão (reta)
    m = -PAR[0] / PAR[1]
    b = -PAR[2] / PAR[1]

    Ye2 = -1 * np.ones(data_treino.shape[1])

    for i in range(0, data_treino.shape[1]):
        if new_data[1, i] - m * new_data[0, i] - b >= 0:
            Ye2[i] = -1


    # Teste do modelo
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(0, new_colunas):
        if transmissao[i] == Ye2[i] and transmissao[i] == 1:  # T=Yest=1
            TP = TP + 1
        if transmissao[i] == Ye2[i] and transmissao[i] == -1:  # T=Yest=0
            TN = TN + 1
        if transmissao[i] == 1 and Ye2[i] == -1:  # T=1, Yes=0
            FN = FN + 1
        if transmissao[i] == -1 and Ye2[i] == 1:  # T=0, Yes=1
            FP = FP + 1

    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    print("\n:::::::::::  FRONTERIA DECISAO ------------------")
    print(" SE - sensibilidade  >", round(SE, 3))
    print(" SP - Especificidade >", round(SP, 3))

knn(data_treino, resul_treino)