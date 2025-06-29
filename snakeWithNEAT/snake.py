import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "100,100"

import pygame
import random as r

GLOBAL_STEP_SIZE = 20
class Snake:
    def __init__(self):
        global GLOBAL_STEP_SIZE
        '''
        body's first element = head
        body's last  element = tail
        '''
        self.dir = r.choice([(0, 1), (0, -1), (-1, 0), (1, 0)])
        self.stepSize = GLOBAL_STEP_SIZE
        self.body = [(240, 240), (240-self.stepSize, 240), (240-self.stepSize*2, 240)]
        self.justAteFood = False


    def step(self):
        if not self.justAteFood:
            self.body.pop()
        else:
            self.justAteFood = False

        newHead = (self.body[0][0]+self.dir[0]*self.stepSize, self.body[0][1]+self.dir[1]*self.stepSize)
        if newHead in self.body:
            return True
        self.body = [newHead] + self.body

        newHeadX, newHeadY = newHead
        if newHeadX <= 0 or newHeadX >= 500 or newHeadY <= 0 or newHeadY >= 500:
            # hit the bounds, game ends
            return True
        else:
            # game ongoing
            return False
    
    def changeDir(self, newDir):
        self.dir = newDir

    def addBodySegment(self):
        self.justAteFood = True

class Apple:
    def __init__(self):
        global GLOBAL_STEP_SIZE
        self.stepSize = GLOBAL_STEP_SIZE
        self.spawn()
        
    def spawn(self):
        self.x, self.y = r.choice(range(self.stepSize, 500+1-self.stepSize, self.stepSize)), r.choice(range(self.stepSize, 500+1-self.stepSize, self.stepSize))

    def step(self, snakeObj:Snake):
        applePos = (self.x, self.y)
        if applePos in snakeObj.body:
            self.spawn()
            snakeObj.addBodySegment()
            return True
        else:
            return False


if __name__ == '__main__':
    pygame.init()

    WIDTH, HEIGHT = (500, 5500)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Snake')

    running = True

    snake = Snake()
    apple = Apple()


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        screen.fill((50, 50, 50))
        for segment in snake.body:
            if segment == snake.body[0]:
                segmentColor = (180, 180, 180)
            else:
                segmentColor = (150, 150, 150)
            pygame.draw.rect(screen, segmentColor, (segment[0], segment[1], snake.stepSize, snake.stepSize)) # x, y, width, height
        pygame.draw.rect(screen, (255, 255, 255), (apple.x, apple.y, apple.stepSize, apple.stepSize)) # x, y, width, height

        snake.step()
        apple.step(snake)
        if r.choice([1, 2]) == 1:
            snake.addBodySegment()

        pygame.display.flip()
        pygame.time.delay(1000)

    pygame.quit()

