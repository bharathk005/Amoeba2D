import aworld as w
import eyes as e

world = w.CreateWorld()

def basic_control():
	world.init_pygame()
	while world.running:
		e.see_amoeba_world(0,50,world.displayX,world.displayY)
		reward = world.run_frame()
		x,y = world.objects[0].body.position
		world.force_dir = world.n_dir


	 
if __name__ == '__main__':
	basic_control()
	print('THE END')
