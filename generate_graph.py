import json
from matplotlib.pyplot import *

with open('data.json') as json_file :
    data  = json.load(json_file)
linsp1 = data[0]
lisnp2 = data[1]
tab_inf = data[2]
for i in range(len(linsp1)) :
    plot(linsp2,tab_inf[i], label = "Proba_de detection : " +str(linsp1[i]))
title('Graphe de d\'infectés en fonction de la probabilité de détection et d\'infection')
xlabel('probabilité d\'infection')
ylabel('nombre d\'infectés')
legend()
show()
savefig('stats.png')
