import aworld as w

def basic_control():
	world = w.CreateWorld()
	world.init_pygame()
	world.run()

if __name__ == '__main__':
	basic_control()
