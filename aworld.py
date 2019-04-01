import random
import pygame
from pygame.key import *
from pygame.locals import *
from pygame.color import *
import pymunk
import pymunk.pygame_util

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
        self.force_dir = 1
        self.force_mag = 5
        self.objects = []
        self.running = True
        self.box_dir = 0
        self.n_dir =1
        self.left_wall = (100.0,100.0)
        self.right_wall = (500.0,100.0)
        self.start_pos = self.left_wall
        self.x_prev = self.start_pos[0]


        self.stacked_frames  = []
        self.stack_size = 4

        self.staticScene()
        self.createBox()
        self.init_pygame()

    def init_stack(self,state):
        self.stacked_frames =  deque([np.zeros(state.shape, dtype=np.int) for i in range(self.stack_size)], maxlen=self.stack_size)

    def stack_states(self,state,reset=0):
        if reset == 1:
            for i in range(self.stack_size):
                self.stacked_frames.append(state)
        elif reset == 0:
            self.stacked_frames.append(state)
        
        stack = np.stack(self.stacked_frames,axis = 2)
        return stack



    def process_frame(self,screen):
        window_dat = pygame.surfarray.array2d(screen)
        offset = 20
        window_dat = window_dat[int(self.left_wall[0]-offset):int(self.right_wall[0]+offset),int(self.displayY-self.left_wall[1]-offset):int(self.displayY-self.right_wall[1]+offset)]
        #window_dat = window_dat.reshape((1,window_dat.shape[0]*window_dat.shape[1]))
        window_dat -= window_dat.min()
        window_dat = window_dat/window_dat.max()
        #state = np1.append(window_dat,self.box_dir)
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
        self.clock = pygame.time.Clock()
        self.screen.blit(font.render("fps: " + str(self.clock.get_fps()), 1, THECOLORS["black"]), (0,0))
        self.screen.blit(font.render("force_dir: " + str(self.force_dir), 1, THECOLORS["black"]), (0,15))
        self.screen.blit(font.render("n_dir: " + str(self.n_dir), 1, THECOLORS["black"]), (0,30))
        self.screen.blit(font.render("Velocity: " + str(velx), 1, THECOLORS["black"]), (0,45))

        state = self.process_frame(self.screen)
        frames = self.stack_states(state)
        pygame.display.flip()

        self.terminate_action()
        # Delay fixed time between frames
        #self.clock.tick(60)
        return reward,frames,self.running
            

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

    def applyForce(self,action):
        obj = self.objects[0]
        obj.body.apply_impulse_at_local_point((self.force_mag*self.action_space[action],0),(0,0))

    def clear_screen(self):
        
        self.screen.fill(THECOLORS["white"])

    def draw_objects(self):
        self.space.debug_draw(self.drawOption)

    def terminate_action(self):
        x,y = self.objects[0].body.position
        if(y < 100.0 ):
            self.running = False
        if self.n_dir == -1 and x <= self.left_wall[0] + 200:
            self.running = False

    def milestones(self):
        x,y = self.objects[0].body.position
        if x >= self.right_wall[0] - 10:
            self.n_dir = -1
        elif x<= self.left_wall[0] +10:
            self.n_dir = 1

    def getReward(self):
        reward = 0
        x,y = self.objects[0].body.position
        x = round(x,4)
        y = round(y,4)
        if self.n_dir == 1:
            if self.x_prev < x:
                reward = 1 
            elif self.x_prev == x:
                reward = 0
            else:
                reward = -1
        elif self.n_dir == -1:
            if self.x_prev > x:
                reward = 1 
            elif self.x_prev == x:
                reward = 0
            else:
                reward = -1
        self.x_prev = x
        return reward


if __name__ == '__main__':
    game = CreateWorld()
    while game.running:
        game.run_frame(1)

    
