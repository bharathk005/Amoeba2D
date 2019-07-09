import random
import pygame
from pygame.key import *
from pygame.locals import *
from pygame.color import *
import pymunk
import pymunk.pygame_util
from collections import deque

import numpy as np1

class CreateWorld(object):
    def __init__(self):
        self.displayX = 600
        self.displayY = 600
        self.space = pymunk.Space()
        self.space.gravity = (0.0,-986)
        self.physicsStepsPerFrame = 1
        self.dt = 1.0/60.0

        self.objects = []
        self.staticBody = self.staticScene()
        self.createObj()
        self.init_pygame()

    def run_frame(self):
        for x in range(self.physicsStepsPerFrame):
            self.space.step(self.dt)
        #self.applyForce()
        self.clear_screen()
        self.draw_objects()

        pygame.display.flip()

    def applyForce(self):
        obj = self.objects[0]
        obj.body.apply_force_at_world_point((-1,0),(0,0))


    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.displayX, self.displayY))
        self.drawOption = pymunk.pygame_util.DrawOptions(self.screen)

    def staticScene(self):
        staticBody = self.space.static_body
        ground = pymunk.Segment(staticBody,(50,100),(250,100),1.5)
        ground.friction = 0.8
        self.space.add(ground)

        ramp = pymunk.Segment(staticBody,(250,100),(550,300),1.5)
        ramp.friction = 0.8
        self.space.add(ramp)
        return staticBody

    def createObj(self):
        mass =5
        inertia = pymunk.moment_for_circle(mass,0, 5)
        body = pymunk.Body(mass,inertia)
        body.position = (75,105)
        shape = pymunk.Circle(body,10)
        shape.elasticity = 0.95
        shape.friction = 0.5
        self.space.add(body,shape)
        self.space.add(pymunk.SimpleMotor(body,self.staticBody,-4))
        self.objects.append(shape)

    def clear_screen(self):
        self.screen.fill(THECOLORS["white"])

    def draw_objects(self):
        self.space.debug_draw(self.drawOption)

if __name__ == '__main__':
    game = CreateWorld()
    while True:
        game.run_frame()