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


# ------------------------------------- Representação grafica das variaveis --------------------------------------------
def grafic_representation():
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

# ================================================ OUTLIERS ============================================================
# ---------------------------Removendo os outliers ja conhecidos em City_mpg--------------------------------------------
# Metodo KNN
def outliers_KNN(tratar):
    k = 3  # numero de vizinhos
    count = 0

    for i in range(lines):
        vizinhos = []
        if tratar[i] > 90:  # Valor 90 escolhido, pois o professor falou que outliers eram > 90
            count += 1

            for j in range(1, k + 1):
                vizinhos.append(tratar[i - k])
                vizinhos.append(tratar[i + k])

            mean = np.mean(vizinhos)
            # print(city_mpg[i], mean, vizinhos)
            tratar[i] = mean

    print("\n ------------------ Outliers ------------------")
    print("Numero de outliers encontrados pelo KNN em 'city_mpg' =", count)


outliers_KNN(new_data[1])  # city_mpg

new_data = np.array(new_data)
new_label = np.array(new_label)

# -------------------------------- Removendo outlier de highway_mph ----------------------------------------------------
# Metodo Filtro através do desvio padrão
def outliers_filter(data_filter, fator):
    mean = np.mean(data_filter)  # MELHOR MÉTODO PARA OUTLIERS (Usa pontos reais)
    desvio = np.std(data_filter)

    limMax = mean + fator * desvio
    limMin = mean - fator * desvio

    outlierMax = np.where(data_filter >= limMax)[0]
    outlierMin = np.where(data_filter <= limMin)[0]

    data_filter[outlierMax] = limMax
    data_filter[outlierMin] = limMin

    soma = np.sum(len(outlierMin) + len(outlierMax))

    print("Numero de outliers encontrados pelo Filter em 'highway_mpg' =", soma)


outliers_filter(new_data[2, :], 3)  # hightway_mpg

# ============================================= PREPARAÇÃO FINAL =======================================================
# ----------------------------------------- Normalização dos dados -----------------------------------------------------
for i in range(new_data.shape[0] - 1):
    new_data[i, :] = new_data[i, :] - new_data[i, :].min()
    new_data[i, :] = new_data[i, :] / new_data[i, :].max()

new_data = np.transpose(new_data)
# -------------------------------- Separando os dados em Treino/Validação ----------------------------------------------
random.shuffle(new_data[0])  # Shuffle apenas nas linhas
data_treino = np.array(new_data[:3574, :3])
resul_treino = np.array(new_data[:3574, 3])

data_validacao = np.array(new_data[3574:, :3])
resul_avaliacao = np.array(new_data[3574:, 3])

# --------------------------------------------- *Criação dos modelos* --------------------------------------------------
# ================================================== Método KNN ========================================================

def NearestKN(P, ignore, data):
    dist = np.empty([np.shape(data)[0], 2])  # Cria matriz dist/idx

    for i in range(np.shape(data)[0]):  # Percorre as linhas
        if i == ignore:
            dist[i] = [9999, i]  # Foi notado que dava distância 0, aqui a gente contorna isso
            continue

        dist[i] = [np.linalg.norm(P - data[i]), i]  # Calculo da distancia

    olddist = dist

    idx = np.argsort(dist[:, 0])  # sort nas distancias
    sortedDist = dist[idx]

    return sortedDist  # Retorna a matriz das distancias filtradas pela mais proxima


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
        neigh_idx = NearestKN(v, i, data_knn)[:, 1]  # Vetor com os índices mais proximos

        nn = []
        for idx in range(k):
            nn.append(neigh_idx[idx])  # Listas com os k indices mais proximos

        nn = np.array(nn)

        sum = 0
        for elem in nn:
            elem = int(elem)
            sum = sum + res_knn[elem]  # Somatório das calssificações dos k vizinhos (0/1)

        xs = data_knn[i, 0]
        ys = data_knn[i, 1]
        zs = data_knn[i, 2]

        if sum > (k / 2):  # Moda
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

    plt.title("Representação grafica da classificação - KNN")
    fig.legend((p0, p1), ('Automático', 'Manual'), loc='upper left')

    ax.set_xlabel("Torque")
    ax.set_ylabel("city_mpg")
    ax.set_zlabel("highway_mpg")
    plt.show()

    # __________________________________________________________________________________________________________________
    SE = TP / (TP + FN)
    SP = TN / (TN + FP)

    PC = TP / (TP + FP)
    F1 = (2*PC*SE) / PC + SE
    print("\n  ------------------ KNN ------------------")
    print(" SE - sensibilidade  =", round(SE, 3))
    print(" SP - Especificidade =", round(SP, 3))
    print(" PC - Precisao =", round(PC, 3))
    print(" F1 - F1Score =", round(F1,3))


# ======================================= Fronteira de Decisão =========================================================

def fronteria_decisao(data_reg, res_reg):
    UM = np.ones((data_reg.shape[0], 1))
    X1 = np.concatenate((data_reg, UM), axis=1)

    LSQ = np.linalg.lstsq(X1, res_reg, rcond=None)
    PAR = LSQ[0]

    ext = np.dot(X1, PAR).round()
    id0 = np.where(ext == 0)[0]
    id1 = np.where(ext == 1)[0]

    FP = 0
    TP = 0
    TN = 0
    FN = 0

    for i in range(0, data_reg.shape[0]):
        if res_reg[i] == ext[i] and res_reg[i] == 1:
            TP = TP + 1
        if res_reg[i] == ext[i] and res_reg[i] == 0:
            TN = TN + 1
        if res_reg[i] == 1 and ext[i] == 0:
            FN = FN + 1
        if res_reg[i] == 0 and ext[i] == 1:
            FP = FP + 1



    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    PC = TP / (TP + FP)
    F1 = (2 * PC * SE) / PC + SE

    print("\n  ------------------ FRONTEIRA DE DECISAO ------------------")
    print(" SE - sensibilidade  =", round(SE, 3))
    print(" SP - Especificidade =", round(SP, 3))
    print(" PC - Precisao =", round(PC, 3))
    print(" F1 - F1Score =", round(F1, 3))
# ============================================== SIMILARIDADE ==========================================================
def similaridade(data_sim, res_sim):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    id0 = np.where(res_sim == 0)[0]
    C0 = np.zeros(3)
    C0[0] = np.mean(data_sim[id0, 0])
    C0[1] = np.mean(data_sim[id0, 1])
    C0[2] = np.mean(data_sim[id0, 2])

    id1 = np.where(res_sim == 1)[0]
    C1 = np.zeros(3)
    C1[0] = np.mean(data_sim[id1, 0])
    C1[1] = np.mean(data_sim[id1, 1])
    C1[2] = np.mean(data_sim[id1, 2])

    Ye1 = np.zeros(data_sim.shape[0])
    for i in range(0, data_sim.shape[0]):
        d0 = np.linalg.norm(C0 - data_sim[i, :])
        d1 = np.linalg.norm(C1 - data_sim[i, :])

        xs = data_sim[i, 0]
        ys = data_sim[i, 1]
        zs = data_sim[i, 2]

        if d1 < d0:  # mais parecido com uma laranja media
            Ye1[i] = 1
            p1 = plt.scatter(xs, ys, zs, 'b')
        else:
            p0 = plt.scatter(xs, ys, zs, 'r')

    plt.title("Representação grafica da classificação - Similaridade")
    fig.legend((p0, p1), ('Automático', 'Manual'), loc='upper left')

    ax.set_xlabel("Torque")
    ax.set_ylabel("city_mpg")
    ax.set_zlabel("highway_mpg")
    plt.show()

    # ----------------------------------------------------------------------------------------------------------------------

    FP = 0
    TP = 0
    TN = 0
    FN = 0
    for i in range(0, data_treino.shape[0]):
        if res_sim[i] == Ye1[i] and res_sim[i] == 1:  # T=Yest=1
            TP = TP + 1
        if res_sim[i] == Ye1[i] and res_sim[i] == 0:  # T=Yest=0
            TN = TN + 1
        if res_sim[i] == 1 and Ye1[i] == 0:  # T=1, Yes=0
            FN = FN + 1
        if res_sim[i] == 0 and Ye1[i] == 1:  # T=0, Yes=1
            FP = FP + 1

    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    PC = TP / (TP + FP)
    F1 = (2 * PC * SE) / PC + SE

    print("\n  ------------------ SIMILARIDADE ------------------")
    print(" SE - sensibilidade  =", round(SE, 3))
    print(" SP - Especificidade =", round(SP, 3))
    print(" PC - Precisao =", round(PC, 3))
    print(" F1 - F1Score =", round(F1, 3))


def indv_rule_class(data_irc, res_irc, var = [0, 1, 2]):
    manual_idx = []
    auto_idx = []

    for idx in range(0, data_irc.shape[0]):
        if res_irc[idx] == 0:
            manual_idx.append(idx)
        else:
            auto_idx.append(idx)

    manual_val = np.array(data_irc[manual_idx][var])
    auto_val = np.array(data_irc[auto_idx][var])

    manual_mean = np.mean(manual_val)
    auto_mean = np.mean(auto_val)

    classifica = []
    for i in range(0, data_irc.shape[0]):
        mean_val = data_irc[i, var]

        dist_manual = np.linalg.norm(mean_val - manual_mean)
        dist_auto = np.linalg.norm(mean_val - auto_mean)

        if dist_manual < dist_auto:
            classifica.append(0)
        else:
            classifica.append(1)

    FP = 0
    TP = 0
    TN = 0
    FN = 0

    for i in range(0, data_treino.shape[0]):
        if res_irc[i] == classifica[i] and res_irc[i] == 1:  # T=Yest=1
            TP = TP + 1
        if res_irc[i] == classifica[i] and res_irc[i] == 0:  # T=Yest=0
            TN = TN + 1
        if res_irc[i] == 1 and classifica[i] == 0:  # T=1, Yes=0
            FN = FN + 1
        if res_irc[i] == 0 and classifica[i] == 1:  # T=0, Yes=1
            FP = FP + 1

    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    PC = TP / (TP + FP)
    F1 = (2 * PC * SE) / PC + SE

    print("\n  ------------------ IRC ------------------")
    print(" SE - sensibilidade  =", round(SE, 3))
    print(" SP - Especificidade =", round(SP, 3))
    print(" PC - Precisao =", round(PC, 3))
    print(" F1 - F1Score =", round(F1, 3))


# KNN(data_treino, resul_treino, 5)
# fronteria_decisao(data_treino, resul_treino)
# similaridade(data_treino, resul_treino)
indv_rule_class(data_treino, resul_treino)
