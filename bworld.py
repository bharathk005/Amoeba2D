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
        self.displayX = 900
        self.displayY = 900
        self.space = pymunk.Space()
        self.space.gravity = (0.0,-986)
        self.physicsStepsPerFrame = 1
        self.dt = 1.0/60.0

        self.objects = []
        self.staticScene()
        self.createCirc((55,100),10,True)
        self.createCirc((125,100),20,True)
        self.createCirc((200,100),10,True)
        self.creatJoint(self.objects[0].body, self.objects[1].body, (0,0), (0,0))
        self.creatJoint(self.objects[1].body, self.objects[2].body, (0,0), (0,0))
        #self.createSpring(self.objects[0].body, self.objects[1].body, (0,0), (0,0),50)
        # self.createSqr()
        #self.createPoly()
        self.init_pygame()

    def run_frame(self):
        for x in range(self.physicsStepsPerFrame):
            self.space.step(self.dt)

        #self.applyForce()
        self.clear_screen()
        self.draw_objects()


        font = pygame.font.SysFont("Arial", 16)
        self.screen.blit(font.render("angularVelocity: " + str(self.objects[0].body.angular_velocity), 1, THECOLORS["black"]), (0,15))
 


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
        ground = pymunk.Segment(staticBody,(50,100),(350,100),1.5)
        ground.friction = 0.8
        self.space.add(ground)

        ramp1 = pymunk.Segment(staticBody,(350,100),(550,180),1.5)
        ramp1.friction = 0.8
        ramp2 = pymunk.Segment(staticBody,(550,180),(700,300),1.5)
        ramp2.friction = 0.8
        self.space.add(ramp1)
        self.space.add(ramp2)
    
    def creatJoint(self,bodyA,bodyB,anchorA,anchorB):
        joint = pymunk.PinJoint(bodyA,bodyB,anchorA,anchorB)
        self.space.add(joint)


    def createSpring(self,bodyA,bodyB,anchorA,anchorB,restLength):
        joint = pymunk.DampedSpring(bodyA,bodyB,anchorA,anchorB,restLength,
            stiffness = 0.5,damping = 0.5)
        self.space.add(joint)

    def createCirc(self,position,radius,motor = False):
        mass =5
        inertia = pymunk.moment_for_circle(mass,0, radius)
        body = pymunk.Body(mass,inertia)
        body.position = position
        shape = pymunk.Circle(body,radius)
        shape.elasticity = 0.95
        shape.friction = 0.5
        self.space.add(body,shape)
        if motor:
            self.space.add(pymunk.SimpleMotor(body,self.space.static_body,-8))
        self.objects.append(shape)

    def createSqr(self,position):
        mass = 5
        inertia = pymunk.moment_for_box(mass,(15,15))
        body = pymunk.Body(mass,inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body,(15,15))
        shape.elasticity = 0.95
        shape.friction = 0.5
        self.space.add(body,shape)
        self.space.add(pymunk.SimpleMotor(body,self.space.static_body,-4))
        self.objects.append(shape)


    def createPoly(self,position):
        mass = 5
        vertices = [(20,0),(0,20),(0,-20)]
        inertia = pymunk.moment_for_poly(mass,vertices)
        body = pymunk.Body(mass,inertia)
        body.position = position
        shape = pymunk.Poly(body,vertices)
        shape.elasticity = 0.95
        shape.friction = 0.5
        self.space.add(body,shape)
        self.space.add(pymunk.SimpleMotor(body,self.space.static_body,-4))
        self.objects.append(shape)

    def clear_screen(self):
        self.screen.fill(THECOLORS["white"])

    def draw_objects(self):
        self.space.debug_draw(self.drawOption)

if __name__ == '__main__':
    game = CreateWorld()
    while True:
        game.run_frame()