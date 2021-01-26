import json
from matplotlib.pyplot import *
from operator import add
with open('data.json') as json_file :
    data  = json.load(json_file)
linsp1 = data[0]
linsp2 = data[1]
tab_inf = data[2]
tab_death = data[3]
tab_restab = data[4]
for i in range(len(linsp1)) :
    plot(linsp2, list( map(add,tab_death[i],list(map(add,tab_inf[i], tab_restab[i])) )), label = "Proba_de detection : " +str(1 - linsp1[i]))
title('Graphe du nombre d\'infectés en fonction de la probabilité de détection et d\'infection')
xlabel('probabilité d\'infection')
ylabel('nombre d\'infectés')
legend()
show()
savefig('stats.png')
