import pygame as pg
import random
import math
import copy

from itertools import count

WIDTH = 700
HEIGHT = 900
ELITISM = 3

class Dot:
    VEL = pg.Vector2((15, 0))  # 3
    RADIUS = 5 # 3
    LIVE_COLOR = 'green'
    DEAD_COLOR = 'gray'
    
    def __init__(self, position, moves=[]):
        self.position = pg.Vector2(position)
        self.directions = moves.copy()
        self.move_idx = 0
        self.alive = True
        
    def reset(self, position):
        self.move_idx = 0
        self.position = position
        self.alive = True
        
    def move(self):
        if self.move_idx < len(self.directions):
            direction = self.directions[self.move_idx]
        else:
            direction = self.VEL.rotate(random.randrange(360))
            self.directions.append(direction)
        
        self.position += direction
        self.move_idx += 1
    
    def collides(self, obstacles):
        for obstacle in obstacles:
            if obstacle.collides(self):
                if isinstance(obstacle, Goal):
                    if False and len(self.directions) - 1 > self.move_idx:
                        self.directions = self.directions[:self.move_idx]
                return True
        
        return False
    
    def draw(self, surface):
        if self.alive:
            pg.draw.circle(surface, self.LIVE_COLOR, self.position, self.RADIUS)
        else:
            pg.draw.circle(surface, self.DEAD_COLOR, self.position, self.RADIUS)
        
    def __str__(self):
        return f'Position:{self.position}\nMoves:{self.move_idx}'

    def __repr__(self):
        return self.__str__()

class Population:
    def __init__(self, position, goal, size=500):
        self.position = position
        self.goal = goal
        self.dots = []
        self.size = size
        self._alive = size
            
    def generate_population(self):
        if len(self.dots) == 0:
            for _ in range(self.size):
                dot = Dot(self.position)
                self.dots.append(dot)
        else:
            new_population = []
            best_dots = self.select_best_dots(ELITISM)
            
            for _ in range(0, self.size - ELITISM, 2):
                parents = random.sample(best_dots, k = 2)
                new_dots = self.crossover(*parents)
                
                self.mutate(new_dots[0])
                self.mutate(new_dots[1])
                
                new_population.extend(new_dots)
            
            for dot in best_dots:
                dot.reset(self.position)
                
            new_population.extend(best_dots)
            
            self._alive = len(new_population)
            self.dots = new_population
            
            return best_dots[0]
            
    def update(self, width, height, obstacles):
        alive = 0
        for dot in self.dots:
            if dot.alive:
                alive += 1
                dot.move()
                
                if dot.position.x < 0 or dot.position.x > width or \
                   dot.position.y < 0 or dot.position.y > height or \
                   dot.collides(obstacles):
                    dot.alive = False
        
        self._alive = alive
        
    def draw(self, surface):
        for dot in self.dots:
            dot.draw(surface)
    
    def select_best_dots(self, n):
        dots_fitness = self.get_dots_fitness()
        
        key = lambda x: x[1]
        dots = sorted(dots_fitness, key=key, reverse=True)
        dots = dots[:n]
        dots = [x[0] for x in dots]
        
        return dots
        
    def get_dot_fitness(self, dot):
        distance_to_goal = math.dist(dot.position, self.goal.rect.center)
        #distance_score = 1 if distance_to_goal <= 5 else (1 / distance_to_goal)
        distance_score = 1000 if distance_to_goal <= 5 else (800 - distance_to_goal)
        fitness = distance_score + (1 / (dot.move_idx))
        return fitness
            
    def get_dots_fitness(self):
        fitness = []
        for dot in self.dots:
            fitness.append((dot, self.get_dot_fitness(dot)))
        
        return fitness
    
    def alive(self):
        return self._alive > 0
    
    def crossover(self, dot1, dot2):
        point = random.randrange(min(dot1.move_idx, dot2.move_idx))
        new_dot1_directions = []
        new_dot2_directions = []
        
        new_dot1_directions.extend(dot1.directions[:point])
        new_dot1_directions.extend(dot2.directions[point:])
        
        new_dot2_directions.extend(dot2.directions[point:])
        new_dot2_directions.extend(dot1.directions[:point])
        
        new_dot1 = Dot(self.position, new_dot1_directions)
        new_dot2 = Dot(self.position, new_dot2_directions)
        
        return new_dot1, new_dot2
        
    def mutate(self, dot, prob=0.01):
        for i in range(len(dot.directions)):
            if random.random() < prob:
                direction = Dot.VEL.rotate(random.randrange(360))
                dot.directions[i] = direction
    
    def __str__(self):
        return f'Alive:{self._alive} Size:{self.size}'
        
        
class Obstacle:
    COLOR = 'black'
    def __init__(self, x, y, width, height):
        self.rect = pg.Rect((x, y, width, height))
        
    def draw(self, surface):
        pg.draw.rect(surface, self.COLOR, self.rect)
    
    def collides(self, dot):
        return self.rect.collidepoint(dot.position)


class Goal(Obstacle):
    COLOR = 'red'
    
   
if __name__ == '__main__':
    window = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption('Dots Simulation')
    clock = pg.time.Clock()
    
    position = (WIDTH // 2, HEIGHT * 0.9)
    goal = Goal(WIDTH // 2, 50, 10, 10)
    
    obstacles = [
        goal,
        Obstacle(WIDTH * 0.75, HEIGHT * 0.25, 100, 20),
        Obstacle(WIDTH * 0.75, HEIGHT * 0.75, 100, 20),
        Obstacle(WIDTH * 0.25, HEIGHT * 0.25, 100, 20),
        Obstacle(WIDTH * 0.25, HEIGHT * 0.75, 100, 20),
        Obstacle(WIDTH * 0.5 - 200, HEIGHT * 0.5 - 10, 400, 20),
    ]
    
    population = Population(position, goal)
    for i in count():
        best = population.generate_population()
        while population.alive():
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    pg.quit()
                    quit()
                    
            population.update(WIDTH, HEIGHT, obstacles)
            
            window.fill('white')
            
            for obstacle in obstacles:
                obstacle.draw(window)
            
            population.draw(window)
            
            pg.display.flip()
            clock.tick(60)
        print('Generation:', i, population)
        print('Best:', population.select_best_dots(50))
        #print('Best:', population.get_best_dots(goal, 100))
        #break