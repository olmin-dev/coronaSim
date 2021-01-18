import sys, math, random
import pygame
import pygame.draw
import numpy as np
import random as rd
import networkx as nx
import matplotlib as plt
import copy
import sys

__screenSize__ = (900,900) #(1280,1280)
__cellSize__ = 10
__gridDim__ = tuple(map(lambda x: int(x/__cellSize__), __screenSize__))
__density__ = 3
__random_covideux__ = 10


if len(sys.argv) > 1:
    __HAS_LEGEND__ = int(sys.argv[1])
else:
    __HAS_LEGEND__ = 1

if len(sys.argv) > 2:
    HAS_QUARANTINE = int(sys.argv[2])
else:
    HAS_QUARANTINE = 0

HEALTHY = 0
UNDETECTED_INFECTED = 1
RESTABLISHED = 2
DEAD = 3
QUARANTINED_HEALTHY = 4
QUARANTINED_INFECTED = 5
QUARANTINED_RESTABLISHED = 6


__colors__ = [(255,255,255),(255,120,0), (0,0,255), (0,0,0), (140,140,140),(200,140,140), (140,140,200)] #sain, infected, restablished, dead, sain quarantine, infected quarantine, restablished quarantined
__colors_corr__ = ["Healty", "Undetected infected", "Restablished", "Dead", "Quarantine healthy", "Quarantine infected", "Quaratine restablished"]

# lambda x : return (x%__gridDim__[0],x//__gridDim__[1])

glidergun=[
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

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

def getColorCell(n):
    return __colors__[n]

class Grid:
    _grid= None
    _gridbis = None
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
        #ones = np.random.random((mx, my)) > 0.75
        #self._grid[nx//2-mx//2:nx//2+mx//2, ny//2-my//2:ny//2+my//2] = ones
        if True:
            for i in range(__random_covideux__) :
                my_cell = 0
                while(my_cell == 0) :
                    i = random.randrange(__gridDim__[0])
                    j = random.randrange(__gridDim__[0])
                    if(self._grid[i][j] == 0) :
                        my_cell = 1
                        self._grid[i][j] = 1
        """
        else: # Else if init with glider gun

            a = np.fliplr(np.rot90(np.array(glidergun),3))
            nx, ny = __gridDim__
            mx, my = a.shape
            self._grid[nx//2-mx//2:nx//2+mx//2, ny//2-my//2:ny//2+my//2] = a
        else :
            self._grid = covid_run
        if False: # True to init with one block at the center
            self._grid[nx//2,ny//2] = 1
            self._grid[nx//2+1,ny//2] = 1
            self._grid[nx//2,ny//2+1] = 1
            self._grid[nx//2+1,ny//2+1] = 1
        elif False: # True to init with random values at the center
            nx, ny = __gridDim__
            mx, my = 20, 16
            ones = np.random.random((mx, my)) > 0.75
            self._grid[nx//2-mx//2:nx//2+mx//2, ny//2-my//2:ny//2+my//2] = ones
            """
    def count_infected(self):
        return np.np.count_nonzero(self._grid == 1)

    def mutation_voisins(self, x, y):
        n = len(self._indexVoisins)
        i = random.randrange(n)
        t = copy.deepcopy(self._indexVoisins)
        x = random.randrange(2)
        s = random.randrange(2)
        y = random.randrange(7)
        a = -3 if s == 0 else 3
        b = y - 3
        t[i] = (a, b) if x == 0 else (b, a)
        return [((dx+x) % 90,(dy+y) % 90) for (dx,dy) in t]

    def indiceVoisins(self, x, y, p):
        if(random.random() < p):
            t = self.mutation_voisins(x, y)
            return t
        return [((dx+x) % 90,(dy+y) % 90) for (dx,dy) in self._indexVoisins]

    def ws_graph(self, n, k, p):
        g = {}
        for c, _ in np.ndenumerate(self._grid):
            voisins = self.indiceVoisins(c[0], c[1], p)
            g[c] = {}
            for v1 in voisins:
                if(v1 in self.__moved_people__.values()) :
                    continue
                g[c][v1] = 1
        return g

    def createMeetings(self):
        self._meeting = self.ws_graph(90,24,0.001)

    def sumEnumerate(self):
        self.createMeetings()
        ret = []
        for c, _ in np.ndenumerate(self._grid):
            tmp = list(self._meeting[c])
            tot = 0
            for i in tmp:
                if self._grid[i[0],i[1]] == 1:
                    tot += 1
            ret.append((c, tot))
            #my_int = sum(list(map(tmp, lambda x : 1 if self._grid._grid[i[0],i[1]] == 1 else 0)))
        return ret

    def make_people_move(self) :
        nb_move = random.randrange(1,6)
        self.__moved_people__ = {}
        for i in range(nb_move) :
            redo = True
            while(redo) :
                line = random.randrange(__gridDim__[0])
                col = random.randrange(__gridDim__[1])
                line1 = random.randrange(__gridDim__[0])
                col1 = random.randrange(__gridDim__[1])
                if((line,col) not in self.__moved_people__.keys() and (self._grid[line,col]<=QUARANTINED_HEALTHY or self._grid[line,col]>=QUARANTINED_RESTABLISHED)) :
                    self.__moved_people__[(line1,col1)] = (line,col)
                    redo = False
        return

    def drawMe(self):
        pass


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
        pygame.font.init() # you have to call this at the start,
        # if you want to use this module.
        self.myfont = pygame.font.SysFont(None,25)

    def drawMe(self):
        if self._grid._grid is None:
            return
        self._screen.fill((255,255,255))
        for x in range(__gridDim__[0]):
            for y in range(__gridDim__[1]):
                pygame.draw.rect(self._screen,
                        getColorCell(self._grid._grid.item((x,y))),
                        (x*__cellSize__ + 1, y*__cellSize__ + 1, __cellSize__-2, __cellSize__-2))
        if (__HAS_LEGEND__ == 1):
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

    def drawText(self, text, position, color = (255,64,64)):
        self._screen.blit(self._font.render(text,1,color),position)

    def cell_evolution(self, c, condition, proba, evolve_if, evolve_else):
        if condition:
            proba_s = random.random()
            if proba_s >= proba:
                self._grid._gridbis[c[0], c[1]] = evolve_if
            else:
                self._grid._gridbis[c[0], c[1]] = evolve_else

    def updateRule(self, infection_proba, survived_proba,detection_proba, action_proba_i, action_proba_d):
        # Maze is B3/S12345
        ''' Many rules in https://www.conwaylife.com/wiki/List_of_Life-like_cellular_automata '''
        self._day += 1
        self._nb_death = 0
        self._nb_infected = 0
        self._nb_restablished = 0
        self._nb_quarantine = 0
        self._nb_quarantine_healthy = 0
        self._grid._gridbis = np.copy(self._grid._grid)
        self._grid.make_people_move()
        for c, s in self._grid.sumEnumerate():
            proba_action = random.random()
            # Death
            self.cell_evolution(c, self._grid._grid[c[0],c[1]] == UNDETECTED_INFECTED and proba_action >= action_proba_d, 1 - survived_proba, RESTABLISHED, DEAD )
            # Infection
            self.cell_evolution(c, self._grid._grid[c[0],c[1]] == HEALTHY and s > 0 and proba_action >= action_proba_i, 1 - (infection_proba)**s, UNDETECTED_INFECTED, HEALTHY )
			# Quarantined death or survived
            self.cell_evolution(c, self._grid._grid[c[0],c[1]] == QUARANTINED_INFECTED and proba_action >= action_proba_d, 1 - survived_proba**(1/2), QUARANTINED_RESTABLISHED, DEAD )
            #Detected
            self.cell_evolution(c, self._grid._grid[c[0],c[1]] == UNDETECTED_INFECTED and self._grid._gridbis[c[0],c[1]] == UNDETECTED_INFECTED and proba_action >= action_proba_i, detection_proba, QUARANTINED_INFECTED, UNDETECTED_INFECTED )
        if HAS_QUARANTINE > 0:
            self.updateQuarantine()
            for c, s in self._grid.sumEnumerate():
                if  self._grid._gridbis[c[0],c[1]] == QUARANTINED_INFECTED:
                    for v in self._grid._meeting[c]:
                        if self._grid._gridbis[v[0], v[1]] == HEALTHY:
                            self._quarantine[v] = 7
                            self._grid._gridbis[v[0], v[1]] = QUARANTINED_HEALTHY
                        if self._grid._gridbis[v[0], v[1]] == UNDETECTED_INFECTED:
                            self._quarantine[v] = 14
                            self._grid._gridbis[v[0], v[1]] = QUARANTINED_INFECTED
                        if self._grid._gridbis[v[0], v[1]] == RESTABLISHED:
                            self._quarantine[v] = 7
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
        for i in list(self._quarantine):
            self._quarantine[i] -= 1
        for i in list(self._quarantine):
            if self._quarantine[i] == 0:
                if self._grid._gridbis[i[0], i[1]] == QUARANTINED_HEALTHY:
                    self._grid._gridbis[i[0], i[1]] = HEALTHY
                if self._grid._gridbis[i[0], i[1]] == QUARANTINED_INFECTED:
                    self._quarantine[i] = 14
                if self._grid._gridbis[i[0], i[1]] == QUARANTINED_RESTABLISHED:
                    self._grid._gridbis[i[0], i[1]] = RESTABLISHED


    def eventClic(self, coord, b):
        pass

    def recordMouseMove(self, coord):
        pass

def main():
    scene = Scene()
    done = False
    clock = pygame.time.Clock()
    while done == False:
        scene.drawMe()
        pygame.display.flip()
        scene.updateRule(0.10, 0.7, 0.95, 0.8, 0.85)
        clock.tick(0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Exiting")
                done=True
    pygame.quit()

if not sys.flags.interactive: main()
