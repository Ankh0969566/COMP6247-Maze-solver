import pygame
import sys

from read_maze import load_maze, get_local_maze_information



# Colors


red = (255, 0, 0)
blue = (0, 0, 255)
darkgreen = (34, 139, 34)
black = (0,0,0)
grey = (192, 192, 192)
yellow = (255,255,0)

Total_Size = 1000, 1000
mazeWH = 1000
origin = (0,0)
grid_width = 2 
maze_shape = 201
cellXY = (mazeWH / maze_shape)



class Maze:
    def __init__(self):


        self.maze = load_maze()
        

        pygame.init()
        self.actor = (1,1)
        self.screen = pygame.display.set_mode(Total_Size)



    def set_dynamic(self, observation, actor, path):
        self.obs = observation
        self.actor = actor
        self.path = path


    def cell_help (self,row,col):
        x = origin[0] + (cellXY * row)+ grid_width / 2
        y = origin[1] + (cellXY * col)+ grid_width / 2
        postion = (x, y, cellXY, cellXY)
        return postion


    def plotCells(self):
        for Row in range(maze_shape):
            for Col in range(maze_shape):

                #wall
                if (self.maze[Col][Row] == 0):
                    postion=self.cell_help(Row,Col)
                    pygame.draw.rect(self.screen, black, postion)
                #End
                if Col == 199 and Row == 199:
                    postion=self.cell_help(Row,Col)
                    pygame.draw.rect(self.screen, yellow, postion)



    def draw_dynamic(self):
        #path
        for s in self.path:
            postion=self.cell_help(s[1],s[0])
            pygame.draw.rect(self.screen, darkgreen, postion)


        for Row in range(3):
            for Col in range(3):
                C = self.actor[0] + (Col - 1)
                R = self.actor[1] + (Row - 1)
                #actor
                if Row == 1 and Col == 1:
                    postion=self.cell_help(R,C)
                    pygame.draw.rect(self.screen, blue, postion)
                #fire
                else:
                    if self.obs[Col][Row][1] > 0:
                        postion=self.cell_help(R,C)
                        pygame.draw.rect(self.screen, red, postion)



    def maze_run(self, observation, actor, path):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.set_dynamic(observation, actor, path)
        self.screen.fill(grey)
        self.plotCells()
        self.draw_dynamic()
        pygame.display.update()