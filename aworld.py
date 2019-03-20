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
        self.space.gravity = (0.0,-90.8)
        self.physicsStepsPerFrame = 1
        self.dt = 1.0/60.0
        self.force_dir = 1
        self.force_mag = 1
        self.objects = []


        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()
        self.drawOption = pymunk.pygame_util.DrawOptions(self.screen)
        self.staticScene()
        self.createBox()
        self.running = True

    def run(self):
        while self.running:
            for x in range(self.physicsStepsPerFrame):
                self.space.step(self.dt)
            self.applyForce()
            self.clear_screen()
            self.draw_objects()
            font = pygame.font.SysFont("Arial", 16)
            self.screen.blit(font.render("fps: " + str(self.clock.get_fps()), 1, THECOLORS["black"]), (0,0))
            self.screen.blit(font.render("force_dir: " + str(self.force_dir), 1, THECOLORS["black"]), (0,15))
            pygame.display.flip()
            # Delay fixed time between frames
            self.clock.tick(50)
            

    def staticScene(self):
        staticBody = self.space.static_body
        ground = pymunk.Segment(staticBody,(100.0, 100.0), (500.0, 100.0), 5.0)
        self.space.add(ground)

    def createBox(self):
        mass = 5
        vertices = (20.0,20.0)
        inertia = pymunk.moment_for_box(mass,(10.0,10.0))
        body = pymunk.Body(mass,inertia)
        body.position = 300.0,150.0
        shape = pymunk.Poly.create_box(body,vertices)
        shape.elasticity = 0.95
        shape.friction = 0.5
        self.space.add(body,shape)
        self.objects.append(shape)

    def applyForce(self):
        obj = self.objects[0]
        obj.body.apply_impulse_at_local_point((self.force_mag*self.force_dir,0))

    def clear_screen(self):
        
        self.screen.fill(THECOLORS["white"])

    def draw_objects(self):
 
        self.space.debug_draw(self.drawOption)

if __name__ == '__main__':
    game = CreateWorld()
    game.run()
