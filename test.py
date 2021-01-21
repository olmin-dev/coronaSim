import sys, math, random
import pygame
import pygame.draw
import numpy as np
import random as rd
import networkx as nx
import matplotlib as plt
import copy
import sys

# Paramètres

__screenSize__ = (900,900)
__cellSize__ = 10
__gridDim__ = tuple(map(lambda x: int(x/__cellSize__), __screenSize__))
__density__ = 3 #inutile pour l'instant
__random_covideux__ = 10 #nombre de personne qui débute la simulation avec la covid

# Entier correspondant à chaque état
HEALTHY = 0
UNDETECTED_INFECTED = 1
RESTABLISHED = 2
DEAD = 3
QUARANTINED_HEALTHY = 4
QUARANTINED_INFECTED = 5
QUARANTINED_RESTABLISHED = 6

# Couleurs
__colors__ = [(255,255,255),(255,120,0), (0,0,255), (0,0,0), (140,140,140),(200,140,140), (140,140,200)] #sain, infected, restablished, dead, sain quarantine, infected quarantine, restablished quarantined
__colors_corr__ = ["Healty", "Undetected infected", "Restablished", "Dead", "Quarantine healthy", "Quarantine infected", "Quaratine restablished"]

# Grilles de départ

covid_run=[
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

# Gestion des paramètres d'entrée
if len(sys.argv) > 1:
    __HAS_LEGEND__ = int(sys.argv[1])
else:
    __HAS_LEGEND__ = 1

if len(sys.argv) > 2:
    HAS_QUARANTINE = int(sys.argv[2])
else:
    HAS_QUARANTINE = 0

def getColorCell(n):
    return __colors__[n]

class Grid:
    _grid= None
    _gridbis = None # Grille temporaire pour la mise à jour des états
    _indexVoisins = [(-2,-2), (-2,-1), (-2, 0), (-2, 1), (-2, 2),(-1,-2), (-1,-1), (-1, 0), (-1, 1), (-1, 2), (0,-2), (0,-1), (0, 1), (0, 2), (1,-2), (1,-1), (1, 0), (1, 1), (1, 2), (2,-2), (2,-1), (2, 0), (2, 1), (2, 2)]
    _indexVoisins_safe = [(-2,-2), (-2,-1), (-2, 0), (-2, 1), (-2, 2),(-1,-2), (-1,-1), (-1, 0), (-1, 1), (-1, 2), (0,-2), (0,-1), (0, 1), (0, 2), (1,-2), (1,-1), (1, 0), (1, 1), (1, 2), (2,-2), (2,-1), (2, 0), (2, 1), (2, 2)]
    _meeting = None
    __moved_people__ = {}

    def __init__(self):
        print("Creating a grid of dimensions " + str(__gridDim__))
        self._grid = np.zeros(__gridDim__, dtype='int8')
        self._gridbis = np.zeros(__gridDim__, dtype='int8')
        nx, ny = __gridDim__
        nx, ny = __gridDim__
        mx, my = 20, 16
        if True: # Génération des infectés
            for i in range(__random_covideux__) :
                my_cell = 0
                while(my_cell == 0) :
                    i = random.randrange(__gridDim__[0])
                    j = random.randrange(__gridDim__[0])
                    if(self._grid[i][j] == 0) :
                        my_cell = 1
                        self._grid[i][j] = 1

    # Modifie le tableau des voisins pour simuler le déplacement de certaines personnes
    def mutation_voisins(self, x, y):
        # On choisit le voisin qui doit changer
        n = len(self._indexVoisins)
        i = random.randrange(n)
        t = copy.deepcopy(self._indexVoisins)
        # On choisit la personne qui va devenir son nouveau voisin
        x = random.randrange(2)
        s = random.randrange(2)
        y = random.randrange(7)
        a = -3 if s == 0 else 3
        b = y - 3
        # On échange une personne dans le cercle d'amis
        t[i] = (a, b) if x == 0 else (b, a)
        #On retourne la liste éditée
        return [((dx+x) % 90,(dy+y) % 90) for (dx,dy) in t]

    def indiceVoisins(self, x, y, p): # Met à jour la liste des voisins pour une case donnée
        if(random.random() < p): # Probabilité de mouvement aléatoire de certaines personnes
            t = self.mutation_voisins(x, y)
            return t
        return [((dx+x) % 90,(dy+y) % 90) for (dx,dy) in self._indexVoisins]

    def ws_graph(self, n, k, p): # Génération du graphe de watts-strogatz
        g = {}
        for c, _ in np.ndenumerate(self._grid):
            voisins = self.indiceVoisins(c[0], c[1], p)
            g[c] = {}
            for v1 in voisins:
                if(v1 in self.__moved_people__.values()) :#les voisins qui ont bougés ne sont plus dans le voisinages
                    continue
                if(v1 in self.__moved_people__.keys()) :# on rajoute ceux qui ont bougé dans le voisinage
                    g[c][self.__moved_people__[v1]] = 1
                    continue
                g[c][v1] = 1
        return g

    def createMeetings(self): # Création du graphe de rencontre
        self._meeting = self.ws_graph(90,24,0.001)

    def sumEnumerate(self): # Compte le nombre de voisins infecté (non mis en quarantane) par case
        self.createMeetings()
        ret = []
        for c, _ in np.ndenumerate(self._grid):
            tmp = list(self._meeting[c])
            tot = 0
            for i in tmp:
                if self._grid[i[0],i[1]] == 1:
                    tot += 1
            ret.append((c, tot))
        return ret

    def make_people_move(self):
        # On choisit le nombre de personne qui vont bouger ce tour
        nb_move = random.randrange(1,6)
        self.__moved_people__ = {}
        for i in range(nb_move) :
            redo = True
            # Tant que l'on a pas trouvé une place qui n'est pas déjà prise et qui n'est pas bloquée par une quarantaine
            while(redo) :
                line = random.randrange(__gridDim__[0])
                col = random.randrange(__gridDim__[1])
                line1 = random.randrange(__gridDim__[0])
                col1 = random.randrange(__gridDim__[1])
                if((line,col) not in self.__moved_people__.keys() and (self._grid[line,col]<=QUARANTINED_HEALTHY or self._grid[line,col]>=QUARANTINED_RESTABLISHED)) :#empêche ceux en quarantaines de bouger
                    # On met à jour notre tableau qui contient les personnes qui ont bougé
                    self.__moved_people__[(line1,col1)] = (line,col)
                    redo = False
        return

class Scene:
    _mouseCoords = (0,0)
    _grid = None
    _font = None
    _quarantine = {}
    _day = 0
    _nb_death = 0
    _nb_infected = 0
    _nb_restablished = 0
    _nb_quarantine = 0
    _nb_quarantine_healthy = 0

    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((__screenSize__[0] + __HAS_LEGEND__ * 250, __screenSize__[1]))
        self._grid = Grid()
        pygame.font.init()
        self.myfont = pygame.font.SysFont(None,25)

    def drawMe(self): # Affichage de la grille
        if self._grid._grid is None:
            return
        self._screen.fill((255,255,255))
        for x in range(__gridDim__[0]):
            for y in range(__gridDim__[1]):
                pygame.draw.rect(self._screen,
                        getColorCell(self._grid._grid.item((x,y))),
                        (x*__cellSize__ + 1, y*__cellSize__ + 1, __cellSize__-2, __cellSize__-2))
        if (__HAS_LEGEND__ == 1): # La jolie légende de Clément
            pygame.draw.rect(self._screen,
                    (0,0,0),
                    (900, 0, 2,900))
            for c, i in enumerate(__colors__):
                pygame.draw.rect(self._screen, i, (91*__cellSize__ - 5, c * __cellSize__ * 3 + 5, __cellSize__ * 3 -2, __cellSize__ * 3-2))
                pygame.draw.rect(self._screen, (0,0,0), (91*__cellSize__ - 5, c * __cellSize__ * 3 + 5, __cellSize__ * 3 -2, __cellSize__ * 3-2), 2)
                textsurface = self.myfont.render(__colors_corr__[c], False, (0, 0, 0))
                self._screen.blit(textsurface,(94 * __cellSize__, c * __cellSize__ * 3 + 10))
            textsurface = self.myfont.render("Day:" + str(self._day), False, (0, 0, 0))
            self._screen.blit(textsurface,(91 * __cellSize__, __screenSize__[0] * 0.98))
            textsurface = self.myfont.render("Dead:" + str(self._nb_death), False, (0, 0, 0))
            self._screen.blit(textsurface,(91 * __cellSize__, __screenSize__[0] * 0.98 - 20))
            textsurface = self.myfont.render("Infected:" + str(self._nb_infected), False, (0, 0, 0))
            self._screen.blit(textsurface,(91 * __cellSize__, __screenSize__[0] * 0.98 - 40))
            textsurface = self.myfont.render("Restablished:" + str(self._nb_restablished), False, (0, 0, 0))
            self._screen.blit(textsurface,(91 * __cellSize__, __screenSize__[0] * 0.98 - 60))
            textsurface = self.myfont.render("Healthy:" + str(__gridDim__[0] ** 2 - self._nb_death - self._nb_infected - self._nb_restablished - self._nb_quarantine_healthy), False, (0, 0, 0))
            self._screen.blit(textsurface,(91 * __cellSize__, __screenSize__[0] * 0.98 - 80))
            textsurface = self.myfont.render("Quarantine and healthy:" + str(self._nb_quarantine_healthy), False, (0, 0, 0))
            self._screen.blit(textsurface,(91 * __cellSize__, __screenSize__[0] * 0.98 - 100))
            textsurface = self.myfont.render("Quarantine:" + str(self._nb_quarantine), False, (0, 0, 0))
            self._screen.blit(textsurface,(91 * __cellSize__, __screenSize__[0] * 0.98 - 120))


    def cell_evolution(self, c, condition, proba, evolve_if, evolve_else): # Fonction de factorisation, qui fait évoluer une cellule d'un état à un autre en fonction d'une proba
        if condition:
            proba_s = random.random()
            if proba_s >= proba:
                self._grid._gridbis[c[0], c[1]] = evolve_if
            else:
                self._grid._gridbis[c[0], c[1]] = evolve_else

    def updateRule(self, infection_proba, survived_proba,detection_proba, action_proba_i, action_proba_d):
        ''' Many rules in https://www.conwaylife.com/wiki/List_of_Life-like_cellular_automata '''
        self._day += 1
        self._nb_death = 0
        self._nb_infected = 0
        self._nb_restablished = 0
        self._nb_quarantine = 0
        self._nb_quarantine_healthy = 0
        self._grid._gridbis = np.copy(self._grid._grid)
        self._grid.make_people_move()
        for c, s in self._grid.sumEnumerate(): #s = le nombre d'infecté autour, c est la cellule
            proba_action = random.random()
            # Death
            self.cell_evolution(c, self._grid._grid[c[0],c[1]] == UNDETECTED_INFECTED and proba_action >= action_proba_d, 1 - survived_proba, RESTABLISHED, DEAD )
            # Infection
            self.cell_evolution(c, self._grid._grid[c[0],c[1]] == HEALTHY and s > 0 and proba_action >= action_proba_i, 1 - (infection_proba)**s, UNDETECTED_INFECTED, HEALTHY )
			# Quarantined death or survived
            self.cell_evolution(c, self._grid._grid[c[0],c[1]] == QUARANTINED_INFECTED and proba_action >= action_proba_d, 1 - survived_proba**(1/2), QUARANTINED_RESTABLISHED, DEAD )
            # Detected
            self.cell_evolution(c, self._grid._grid[c[0],c[1]] == UNDETECTED_INFECTED and self._grid._gridbis[c[0],c[1]] == UNDETECTED_INFECTED and proba_action >= action_proba_i, detection_proba, QUARANTINED_INFECTED, UNDETECTED_INFECTED )
        if HAS_QUARANTINE > 0:
            self.updateQuarantine()
            for c, s in self._grid.sumEnumerate():
                if  self._grid._gridbis[c[0],c[1]] == QUARANTINED_INFECTED:
                    for v in self._grid._meeting[c]:
                        if self._grid._gridbis[v[0], v[1]] == HEALTHY:
                            self._quarantine[v] = 7 #7 jour de quarantaine
                            self._grid._gridbis[v[0], v[1]] = QUARANTINED_HEALTHY
                        if self._grid._gridbis[v[0], v[1]] == UNDETECTED_INFECTED:
                            self._quarantine[v] = 14 #14 jour de quarantaine
                            self._grid._gridbis[v[0], v[1]] = QUARANTINED_INFECTED
                        if self._grid._gridbis[v[0], v[1]] == RESTABLISHED:
                            self._quarantine[v] = 7 #7 jour de quarantaine
                            self._grid._gridbis[v[0], v[1]] = QUARANTINED_RESTABLISHED
        for c, s in self._grid.sumEnumerate():
            if self._grid._gridbis[c[0],c[1]] == DEAD:
                self._nb_death += 1
            if self._grid._gridbis[c[0],c[1]] == RESTABLISHED:
                self._nb_restablished += 1
            if self._grid._gridbis[c[0],c[1]] == QUARANTINED_INFECTED:
                self._nb_quarantine += 1
                self._nb_infected += 1
            if self._grid._gridbis[c[0],c[1]] == QUARANTINED_HEALTHY:
                self._nb_quarantine += 1
                self._nb_quarantine_healthy += 1
            if self._grid._gridbis[c[0],c[1]] == QUARANTINED_RESTABLISHED:
                self._nb_quarantine += 1
                self._nb_restablished += 1
            if self._grid._gridbis[c[0],c[1]] == UNDETECTED_INFECTED:
                self._nb_infected += 1

        self._grid._grid = np.copy(self._grid._gridbis)

    def updateQuarantine(self):
        for i in list(self._quarantine): # Met fin à la quarantaine si le temps est écoulé
            self._quarantine[i] -= 1
        for i in list(self._quarantine):
            if self._quarantine[i] == 0:
                if self._grid._gridbis[i[0], i[1]] == QUARANTINED_HEALTHY:
                    self._grid._gridbis[i[0], i[1]] = HEALTHY
                if self._grid._gridbis[i[0], i[1]] == QUARANTINED_INFECTED:
                    self._quarantine[i] = 14
                if self._grid._gridbis[i[0], i[1]] == QUARANTINED_RESTABLISHED:
                    self._grid._gridbis[i[0], i[1]] = RESTABLISHED

def main():
    scene = Scene()
    done = False
    clock = pygame.time.Clock()
    while done == False:
        scene.drawMe()
        pygame.display.flip()
        scene.updateRule(0.10, 0.7, 0.7, 0.8, 0.85)
        clock.tick(0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Exiting")
                done=True
    pygame.quit()

if not sys.flags.interactive: main()

"""
# TODO:

-> Infecter les gens qui voyagent XX
-> Les vieux et les jeunes
-> Stats en fonction des probas
-> Readme pour expliquer les features
"""
