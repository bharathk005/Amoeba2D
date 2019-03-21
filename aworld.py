import random
import pygame
from pygame.key import *
from pygame.locals import *
from pygame.color import *
import pymunk
import pymunk.pygame_util

class CreateWorld(object):
    def __init__(self):

        self.space = pymunk.Space()
        self.space.gravity = (0.0,-986)

        self.physicsStepsPerFrame = 1
        self.dt = 1.0/60.0
        self.force_dir = 1
        self.force_mag = 5
        self.objects = []
        self.running = True

        self.n_dir =1
        self.left_wall = (200.0,100.0)
        self.right_wall = (400.0,100.0)
        self.start_pos = (201.0,101.0)
        self.x_prev = self.start_pos[0]

        self.staticScene()
        self.createBox()

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()
        self.drawOption = pymunk.pygame_util.DrawOptions(self.screen)

    def run_frame(self):       
        for x in range(self.physicsStepsPerFrame):
            self.space.step(self.dt)
        velx,vely = self.objects[0].body.velocity_at_local_point((0,0))
        self.milestones()
        self.applyForce()
        reward = self.getReward()
        self.clear_screen()
        self.draw_objects()
        font = pygame.font.SysFont("Arial", 16)
        self.screen.blit(font.render("fps: " + str(self.clock.get_fps()), 1, THECOLORS["black"]), (0,0))
        self.screen.blit(font.render("force_dir: " + str(self.force_dir), 1, THECOLORS["black"]), (0,15))
        self.screen.blit(font.render("n_dir: " + str(self.n_dir), 1, THECOLORS["black"]), (0,30))
        self.screen.blit(font.render("Velocity: " + str(velx), 1, THECOLORS["black"]), (0,45))
        pygame.display.flip()
        self.terminate_action()
        # Delay fixed time between frames
        self.clock.tick(60)
        return reward
            

    def staticScene(self):
        staticBody = self.space.static_body
        ground = pymunk.Segment(staticBody,(50.0, 100.0), (550.0, 100.0), 2.0)
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

    def applyForce(self):
        obj = self.objects[0]
        obj.body.apply_impulse_at_local_point((self.force_mag*self.force_dir,0),(0,0))

    def clear_screen(self):
        
        self.screen.fill(THECOLORS["white"])

    def draw_objects(self):
 
        self.space.debug_draw(self.drawOption)

    def terminate_action(self):
        x,y = self.objects[0].body.position
        if(y < 100.0 ):
            self.running = False

    def milestones(self):
        x,y = self.objects[0].body.position
        if x >= self.right_wall[0]:
            self.n_dir = -1
        elif x<= self.left_wall[0]:
            self.n_dir = 1

    def getReward(self):
        reward = 0
        x,y = self.objects[0].body.position
        if self.n_dir == 1:
            if self.x_prev < x:
                reward = 1 
            else:
                reward = 0
        elif self.n_dir == -1:
            if self.x_prev > x:
                reward = 1 
            else:
                reward = 0
        self.x_prev = x
        print(reward)
        return reward 


if __name__ == '__main__':
    game = CreateWorld()
    game.init_pygame()
    while game.running:
        game.run_frame()

    
