import aworld as w
import eyes as e
importmport pygame

from matplotlib import pyplot as plt

world = w.CreateWorld()

#new comm
def basic_control():
	while world.running:

		#e.see_amoeba_world(0,50,world.displayX,world.displayY)
		reward,window_dat,box_dir = world.run_frame()
		x,y = world.objects[0].body.position
		world.force_dir = world.n_dir 
		

		# temp = pygame.display.set_mode((world.displayX, world.displayY))
		# temp.blit(pygame.surfarray.make_surface(window_dat),(0,0) )
		# pygame.display.update()
		# if x > 450 and world.n_dir == 1:
		# 	plt.imshow(window_dat)
		# 	plt.show()




	 
if __name__ == '__main__':
	basic_control()
	print('THE END')
