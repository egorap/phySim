import cv2, math
import numpy as np

def display_liniar_animation():
	for i in range(10, 110):
		img=np.zeros((1000, 1000, 3))
		cv2.circle(img, (i, 250), 1, (0, 0, 255), -1)
		cv2.imshow('Liniar Animation', img)
		cv2.waitKey(1)
	cv2.destroyAllWindows()

def display_function():
	img=np.zeros((1000,1000, 3))
	cv2.line(img, (0,100), (1000, 100), (255,255,255), 1)
	while True:
		for x in range(0,1000):
			y1=round(math.cos(x/50)*50)
			y2=round(math.sin(x/50)*50)
			y3=y1+y2

			cv2.circle(img, (x, y1+100), 1, (0,0,255), -1)
			cv2.circle(img, (x, y2+100), 1, (0,255,0), -1)
			cv2.circle(img, (x, y3+100), 1, (255,0,0), -1)

			img_copy = img.copy()
			cv2.circle(img_copy, (x, y1+100), 3, (0,0,255), -1)
			cv2.circle(img_copy, (x, y2+100), 3, (0,255,0), -1)
			cv2.circle(img_copy, (x, y3+100), 3, (255,0,0), -1)

			cv2.imshow('Trig Functions', img_copy)
			key=cv2.waitKey(1)
			if key != -1: return

def display_function_state_based():
	# rather than duing computation every frame
	# get the function first and than show it (no lag for big computations)
	state=[]
	for t in range(1000):
		y1=round(math.cos(t/50)*50)
		y2=round(math.sin(t/50)*50)
		y3=y1+y2
		state.append([(t,y1), (t,y2), (t,y3)])

	img=np.zeros((1000,1000, 3))
	cv2.line(img, (0,100), (1000, 100), (255,255,255), 1)
	while True:
		for (x,y1),(x,y2),(x,y3) in state:
			cv2.circle(img, (x, y1+100), 1, (0,0,255), -1)
			cv2.circle(img, (x, y2+100), 1, (0,255,0), -1)
			cv2.circle(img, (x, y3+100), 1, (255,0,0), -1)

			img_copy = img.copy()
			cv2.circle(img_copy, (x, y1+100), 3, (0,0,255), -1)
			cv2.circle(img_copy, (x, y2+100), 3, (0,255,0), -1)
			cv2.circle(img_copy, (x, y3+100), 3, (255,0,0), -1)

			cv2.imshow('Trig Functions', img_copy)
			key=cv2.waitKey(1)
			if key == ord('r'):
				print('r')
				break
			elif key != -1:
				print(key)
				return

def display_two_particles():
	#state
	# G=G = 6.67430e-11 # meters and kg
	G=6.67430e-8 # cm and grams

	state=np.array(())
	a_vectors=np.array(())
	p1=np.array((100, 255), dtype=np.float64)
	p2=np.array((300, 400), dtype=np.float64)
	v1=np.array((0.001, 0.005), dtype=np.float64)
	v2=np.array((-0.2, -0.6), dtype=np.float64)
	a1=np.array((0, 0), dtype=np.float64)
	a2=np.array((0, 0), dtype=np.float64)
	m1=1000000000 # gram
	m2=100000000

	state=np.append(state, [p1, p2])
	sim_time=10000*2
	print(f'simulating {sim_time/60} minutes')
	print(f'simulating {sim_time/60/60} hours')
	for i in range(sim_time): # 1 frame = 1 second
		dist=np.linalg.norm(p2-p1)
		dir_vecot=(p2-p1)/dist
		f = (G*m1*m2)/dist**2

		a1=(f/m1)*dir_vecot
		v1+=a1
		p1+=v1

		# print('\ni:', i)

		# print(f)
		a2=(f/m2)*-dir_vecot
		# print(f/m2)
		# print(-dir_vecot)
		# print(a2)
		# print(v2)
		v2+=a2
		# print(v2)
		# print(p2)
		p2+=v2
		# print(p2)

		# if i ==5:
		# 	exit()

		if i%10==0: 
			state=np.append(state, [p1, p2])
			a_vectors=np.append(a_vectors, [a1, a2])

	state=state.astype(int).reshape(-1,2,2)+(1000,500)
	a_vectors=(a_vectors*10**6).astype(int).reshape(-1,2,2)
	# print(state)
	#view
	while True:
		trace=np.zeros((2000, 2000, 3))
		for (p1, p2), (v1, v2) in zip(state, a_vectors):

			ratio=m1/m2

			cv2.circle(trace, p1, 1, (0,0,255), -1)
			cv2.circle(trace, p2, 1, (255,0,0), -1)

			img=trace.copy()

			cv2.circle(img, p1, max(3, 10), (0,0,255), -1)
			cv2.circle(img, p2, max(3, 10), (255,0,0), -1)

			cv2.line(img, p1,v1+p1, (0,0,255), 1)
			cv2.line(img, p2,v2+p2, (255,0,0), 1)

			cv2.imshow('Particals Sim', img)
			key=cv2.waitKey(1)
			if key == ord('r'):
				print('r')
				break
			elif key != -1:
				print(key)
				return

def display_two_particles_m1_centered():
	#state
	# G=G = 6.67430e-11 # meters and kg
	G=6.67430e-8 # cm and grams

	state=np.array(())
	a_vectors=np.array(())
	p1=np.array((100, 255), dtype=np.float64)
	p2=np.array((500, 500), dtype=np.float64)
	v1=np.array((0.001, 0.005), dtype=np.float64)
	v2=np.array((-0.2, 0.5), dtype=np.float64)
	a1=np.array((0, 0), dtype=np.float64)
	a2=np.array((0, 0), dtype=np.float64)
	m1=10000000000 # gram
	m2=100000000

	state=np.append(state, [p1, p2])
	sim_time=10000*2
	print(f'simulating {sim_time/60} minutes')
	print(f'simulating {sim_time/60/60} hours')
	for i in range(sim_time): # 1 frame = 1 second
		dist=np.linalg.norm(p2-p1)
		dir_vecot=(p2-p1)/dist
		f = (G*m1*m2)/dist**2

		a1=(f/m1)*dir_vecot
		v1+=a1
		p1+=v1

		# print('\ni:', i)

		# print(f)
		a2=(f/m2)*-dir_vecot
		# print(f/m2)
		# print(-dir_vecot)
		# print(a2)
		# print(v2)
		v2+=a2
		# print(v2)
		# print(p2)
		p2+=v2
		# print(p2)

		# if i ==5:
		# 	exit()

		if i%10==0: 
			state=np.append(state, [p1, p2])
			a_vectors=np.append(a_vectors, [a1, a2])

	state=state.astype(int).reshape(-1,2,2)+(1000,500)
	a_vectors=(a_vectors*10**6).astype(int).reshape(-1,2,2)
	# print(state)
	#view
	while True:
		trace=np.zeros((2000, 2000, 3))
		for (p1, p2), (v1, v2) in zip(state, a_vectors):

			p2=p2-p1+(1000,500)
			p1=(1000,500)

			ratio=m1/m2

			cv2.circle(trace, p1, 1, (0,0,255), -1)
			cv2.circle(trace, p2, 1, (255,0,0), -1)

			img=trace.copy()

			cv2.circle(img, p1, max(3, 10), (0,0,255), -1)
			cv2.circle(img, p2, max(3, 10), (255,0,0), -1)

			cv2.line(img, p1,v1+p1, (0,0,255), 1)
			cv2.line(img, p2,v2+p2, (255,0,0), 1)

			cv2.imshow('Particals Sim', img)
			key=cv2.waitKey(1)
			if key == ord('r'):
				print('r')
				break
			elif key != -1:
				print(key)
				return




if __name__=="__main__":
	# display_liniar_animation()
	# display_function()
	# display_function_state_based()
	# display_two_particles()
	display_two_particles_m1_centered()
