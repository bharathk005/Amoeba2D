import aworld as w

def basic_control():
	world = w.CreateWorld()
	world.init_pygame()
	while world.running:
		reward = world.run_frame()
		x,y = world.objects[0].body.position
		world.force_dir = world.n_dir
		

	 
if __name__ == '__main__':
	basic_control()
	print('THE END')
