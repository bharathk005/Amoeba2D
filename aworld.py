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
        self.action_size = 2
        self.action_space = [-1 , 1]
        self.space = pymunk.Space()
        self.space.gravity = (0.0,-986)
        self.physicsStepsPerFrame = 1
        self.dt = 1.0/60.0
        self.force_dir = 0
        self.force_mag = 5
        self.objects = []
        self.trash = []
        self.running = True
        self.box_dir = 0
        self.n_dir =1
        self.left_wall = (50.0,100.0)
        self.right_wall = (550.0,100.0)
        self.start_pos = ((self.left_wall[0]+self.right_wall[0])/2 ,self.left_wall[1])
        self.x_prev = self.start_pos[0]

        self.trashpos = (np1.random.uniform(self.left_wall[0] + 50, self.right_wall[0] - 50),self.left_wall[1] + 10)

        self.stacked_frames  = []
        self.stack_size = 3

        self.renderSC = True
        self.staticScene()
        self.createBox()
        self.createTrash()
        self.init_pygame()

    def init_stack(self,state):
        self.stacked_frames =  deque([np1.zeros(state.shape, dtype=np1.int) for i in range(self.stack_size)], maxlen=self.stack_size)

    def stack_states(self,state,reset=0):
        if reset == 1:
            for i in range(self.stack_size):
                self.stacked_frames.append(state)
        elif reset == 0:
            self.stacked_frames.append(state)
        
        stack = np1.stack(self.stacked_frames, axis =2)
        return stack



    def process_frame(self,screen):
        window_dat = pygame.surfarray.array2d(screen)
        offsetx = 20
        offsety = 50
        window_dat = window_dat[int(self.left_wall[0]-offsetx):int(self.right_wall[0]+offsetx),int(self.displayY-self.left_wall[1]-offsety):int(self.displayY-self.right_wall[1]+offsety)]

        window_dat -= window_dat.min()
        window_dat = window_dat/window_dat.max()

        return window_dat

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.displayX, self.displayY))
        self.drawOption = pymunk.pygame_util.DrawOptions(self.screen)

    def reset(self):
        self.__init__()
        self.clear_screen()
        self.draw_objects()
        state = self.process_frame(self.screen)
        self.init_stack(state)
        frames = self.stack_states(state,reset = 1)

        return frames

    def render(self,rend = True):
        self.renderSC = rend

    def run_frame(self, action):       
        for x in range(self.physicsStepsPerFrame):
            self.space.step(self.dt)
        velx,vely = self.objects[0].body.velocity_at_local_point((0,0))
        self.milestones()
        self.applyForce(action)
        x,y = self.objects[0].body.position

        x = round(x,4)
        
        if x - self.x_prev > 0:
            self.box_dir =1
        elif x - self.x_prev == 0:
            self.box_dir = 0
        else:
         self.box_dir = -1 
        reward = self.getReward()
            
        self.clear_screen()
        self.draw_objects()
        font = pygame.font.SysFont("Arial", 16)
        self.screen.blit(font.render("force_dir: " + str(self.action_space[action]), 1, THECOLORS["black"]), (0,15))
        self.screen.blit(font.render("n_dir: " + str(self.n_dir), 1, THECOLORS["black"]), (0,30))
        self.screen.blit(font.render("Velocity: " + str(velx), 1, THECOLORS["black"]), (0,45))

        state = self.process_frame(self.screen)
        frames = self.stack_states(state)
        if self.renderSC:
            pygame.display.flip()

        self.terminate_action()
        # Delay fixed time between frames
        #self.clock.tick(60)
        return reward,frames,not self.running
            

    def staticScene(self):
        staticBody = self.space.static_body
        ground = pymunk.Segment(staticBody,self.left_wall, self.right_wall, 1.5)
        ground.friction = 0.6
        self.space.add(ground)

    def createBox(self):
        mass = 1
        vertices = (10.0,10.0)
        inertia = pymunk.moment_for_box(mass,(10.0,10.0))
        body = pymunk.Body(mass,inertia)
        body.position = self.start_pos
        shape = pymunk.Poly.create_box(body,vertices)
        shape.elasticity = 0.95
        shape.friction = 0.5
        self.space.add(body,shape)
        self.objects.append(shape)

    def createTrash(self):
        mass = 1
        intertia = pymunk.moment_for_circle(mass, 0, 5)
        body = pymunk.Body(mass, intertia)
        self.trashpos = (np1.random.uniform(self.left_wall[0] + 50, self.right_wall[0] - 50),self.left_wall[1] + 10)
        body.position = self.trashpos
        shape = pymunk.Circle(body, 10)
        shape.elasticity = 0.95
        shape.friction = 0.1
        self.space.add(body,shape)
        self.trash.append(shape)

    def delTrashObj(self):
        self.space.remove(self.trash[0])
        self.trash.clear()
        self.createTrash()

    def applyForce(self,action):
        obj = self.objects[0]
        obj.body.apply_impulse_at_local_point((self.force_mag*self.action_space[action],0),(0,0))

    def clear_screen(self):
        
        self.screen.fill(THECOLORS["white"])

    def draw_objects(self):
        self.space.debug_draw(self.drawOption)

    def terminate_action(self):
        x,y = self.objects[0].body.position
        if y < self.left_wall[1] - 10 :
            self.running = False


    def milestones(self):
        x,y = self.objects[0].body.position
        tx,ty = self.trash[0].body.position
        if x >= tx:
            self.n_dir = -1
        elif x < tx:
            self.n_dir = 1

    def getReward(self):
        reward = 0
        x,y = self.objects[0].body.position
        tx,ty = self.trash[0].body.position
        x = round(x,4)
        y = round(y,4)
        if self.n_dir == 1:
            if self.x_prev + 1 < x:
                reward = 0.5
                self.x_prev = x
            elif self.x_prev == x:
                reward = 0
            else:
                reward = 0
        elif self.n_dir == -1:
            if self.x_prev - 1 > x:
                reward = 0.5
                self.x_prev = x
            elif self.x_prev == x:
                reward = 0
            else:
                reward = 0
        
        if ty < self.left_wall[1] - 10:
            reward += 2
            self.delTrashObj()

        if x > self.right_wall[0] or x < self.left_wall[0]:
            reward += 0 #-10

        return reward


if __name__ == '__main__':
    game = CreateWorld()
    while game.running:
        game.run_frame(1)

    
