from matplotlib import pyplot as plt

# definição da paleta de cores
cmap = plt.get_cmap('Greys')
gray_9 = cmap(0.9)
gray_8 = cmap(0.8)
gray_42 = cmap(0.42)

intensidade = 0.95
# Tebet
cmap = plt.get_cmap('YlOrBr')
c_tebet = cmap(intensidade)
c_tebet1 = cmap(0.5)

# Lula
cmap = plt.get_cmap('Reds')
c_lula = cmap(intensidade)

# Bolsonaro
cmap = plt.get_cmap('Greens')
c_bolsonaro = cmap(intensidade)

# Ciro
cmap = plt.get_cmap('Blues')
c_ciro = cmap(intensidade)