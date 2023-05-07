import numpy as np

dataset = np.load('projeto_dataset.npy')
names_dataset = np.load('names_dataset.npy')

colunas = dataset.shape[1]
lines = dataset.shape[0]

# altura = dataset[:,0]
# comprimento = dataset[:,1]
# largura = dataset[:,2]
# potencia = dataset[:,3]
# torque = dataset[:,4]
# hybrid = dataset[:,5] #Motor eletrico ou combustao

# transmissao = dataset[:,6] #missing data

# tracao = dataset[:,7]
# nb_gears = dataset[:,8]
# fabricante = dataset[:,9]
# city_mpg = dataset[:,10]
# highway_mpg = dataset[:,11]
# ano = dataset[:,12]
# fuel_type = dataset[:,13]

print(dataset.shape)

CC_List = []
for col in range(colunas):
    CC_List.append(np.ma.corrcoef(dataset[:, col], dataset[:, 6])[0, 1])

CC_List[5] = np.ma.getmask(CC_List[5])
