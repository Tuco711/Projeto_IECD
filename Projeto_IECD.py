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
            # print(f'{row}={data[:,col][row]}')
            plt.scatter(row, data[:, col][row])


# -------------------------- Verificando se o VOLUME do motor tem alguma correlação significativa ----------------------
def vol_motor():
    vol_motor = []
    for i in range(lines):
        vol = data[i, 0] * data[i, 1] * data[i, 2]
        vol_motor.append(vol)

    corre_motor = np.corrcoef(vol_motor, transmissao)[0][1]
    print(corre_motor)

    # O volume do motor não tem relação significativa com a transmissão


# --------------------------------------- Seleção de variaveis ---------------------------------------------------------
def cc_filter(val1, val2):
    lst = []
    for idx in range(len(CC_list)):
        if val1 < CC_list[idx] < val2:
            lst.append(idx)
    return lst

CCs = cc_filter(0.20, 0.9)

new_data = []
new_label = []

for col in CCs:
    new_data.append(data[:, col])  # Esse passo transpõe os dados
    new_label.append(labels[col])  # Linha vira coluna, coluna vira linha

new_data.append(transmissao)

# ---------------------------Removendo os outliers ja conhecidos em City_mpg--------------------------------------------
# Metodo KNN
def outliers_KNN(variavel):
    tratar = variavel                       # Como usa as médias, pode criar pontos que não existem
    k = 3  # numero de vizinhos

    for i in range(lines):
        vizinhos = []
        if tratar[i] > 90:  # Valor 90 escolhido, pois o professor falou que outliers eram > 90

            for j in range(1, k + 1):
                vizinhos.append(tratar[i - k])
                vizinhos.append(tratar[i + k])

            mean = np.mean(vizinhos)
            # print(city_mpg[i], mean, vizinhos)
            tratar[i] = mean

    variavel = tratar

outliers_KNN(new_data[1])  # city_mpg

new_data = np.array(new_data)
new_label = np.array(new_label)

# -------------------------------- Removendo outlier de highway_mph ----------------------------------------------------
# Metodo Filtro através do desvio padrão
def outliers_filter(data_filter, fator):
    mean = np.mean(data_filter)                 # MELHOR MÉTODO PARA OUTLIERS (Usa pontos reais)
    desvio = np.std(data_filter)

    limMax = mean + fator * desvio
    limMin = mean - fator * desvio

    outlierMax = np.where(data_filter >= limMax)[0]
    outlierMin = np.where(data_filter <= limMin)[0]

    data_filter[outlierMax] = limMax
    data_filter[outlierMin] = limMin

outliers_filter(new_data[2, :], 2)  # hightway_mpg
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
data_treino = np.transpose(data_treino)
resul_treino = np.transpose(resul_treino)

data_validacao = np.array(new_data[:3, 3574:])
resul_avaliacao = np.array(new_data[3, 3574:])
data_validacao = np.transpose(data_validacao)
resul_avaliacao = np.transpose(resul_avaliacao)

# --------------------------------------------- Criação dos modelos ----------------------------------------------------

# ================================================== Método KNN ========================================================

def NearestKN(P, ignore, data):
    dist = np.empty([np.shape(data)[0], 2])     # Cria matriz dist/idx

    for i in range(np.shape(data)[0]):      # Percorre as linhas
        if i == ignore:
            dist[i] = [9999, i]     # Foi notado que dava distância 0, aqui a gente contorna isso
            continue

        dist[i] = [np.linalg.norm(P - data[i]), i]      # Calculo da distancia

    olddist = dist

    idx = np.argsort(dist[:, 0])    # sort nas distancias
    sortedDist = dist[idx]

    return sortedDist       # Retorna a matriz das distancias filtradas pela mais proxima


def KNN(data_knn, res_knn, k):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(np.shape(data_knn)[0]):
        classifica = 0

        v = data_knn[i]
        neigh_idx = NearestKN(v, i, data_knn)[:, 1]     # Vetor com os índices mais proximos

        nn = []
        for idx in range(k):
            nn.append(neigh_idx[idx])   # Listas com os k indices mais proximos

        nn = np.array(nn)

        sum = 0
        for elem in nn:
            elem = int(elem)
            sum = sum + res_knn[elem]   # Somatório das calssificações dos k vizinhos (0/1)


        xs = data_knn[i, 0]
        ys = data_knn[i, 1]
        zs = data_knn[i, 2]

        if sum > (k/2):     # Moda
            p0 = plt.scatter(xs, ys, zs, 'b')
            classifica = 1

        else:
            p1 = plt.scatter(xs, ys, zs, 'r')
            classifica = 0

        # Calculo de avaliação do modelo
        if classifica == res_knn[i] and classifica == 1:
            TP += 1
        if classifica == res_knn[i] and classifica == 0:
            TN += 1
        if classifica == 1 and res_knn[i] == 0:
            FN += 1
        if classifica == 0 and res_knn[i] == 1:
            FP += 1

    plt.title("Representação grafica da classificação")
    fig.legend((p0, p1), ('Elétrico', 'Combustão'), loc='upper left')

    ax.set_xlabel("Torque")
    ax.set_ylabel("city_mpg")
    ax.set_zlabel("highway_mpg")
    plt.show()
# ______________________________________________________________________________________________________________________
    SE=TP/(TP+FN)
    SP=TN/(TN+FP)
    print("\n  ------------------ KNN ------------------")
    print(" SE - sensibilidade  =", round(SE,3))
    print(" SP - Especificidade =", round(SP,3))

# ======================================= Fronteira de Decisão =========================================================

def fronteria_decisao():
    # Calculo dos Minimos quadrados
    UM = np.ones((data_treino.shape[1], 3574))
    X1 = np.concatenate((data_treino, UM), axis=0)
    id0 = np.where(resul_treino == 0)[0]
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


KNN(data_validacao, resul_avaliacao, 5)
