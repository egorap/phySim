import cv2, math
import numpy as np

# G=6.67430e-11 # meters and kg
G=6.67430e-8 # cm and grams

def get_acceleration(p1, p2):
	dist=np.linalg.norm(p2[0]-p1[0])
	return (G*p2[3][0])*(p2[0]-p1[0])/dist**3

def view_simulation(state, a_vectors):
	import cv2, numpy as np
	colors = [(0,0,255), (255,0,0), (0,255,0),
	          (255,255,0), (255,0,255), (0,255,255)]
	num_frames = len(state)
	
	while True:
		for i in range(num_frames):
			img = np.zeros((1000, 1000, 3), dtype=np.uint8)
			curr_pos = state[i]    # shape: (n,2)
			curr_acc = a_vectors[i] # shape: (n,2)
			
			# Determine bounding box and scale factor
			min_pos = np.min(curr_pos, axis=0)
			max_pos = np.max(curr_pos, axis=0)
			box_size = np.max(max_pos - min_pos)
			margin = 150  # pixels
			if box_size > 0:
				scale = (1000 - 2*margin) / box_size
			else:
				scale = 1.0
			center = (min_pos + max_pos) / 2.0
			offset = np.array([500, 500]) - center*scale
			
			# Optionally draw grid (scaled)
			grid_spacing = 100
			scaled_spacing = int(grid_spacing * scale)
			if scaled_spacing != 0:
				for x in range(-1000, 2000, scaled_spacing):
					pt1 = (x + int(offset[0]), 0)
					pt2 = (x + int(offset[0]), 1000)
					cv2.line(img, pt1, pt2, (50,50,50), 1)
				for y in range(-1000, 2000, scaled_spacing):
					pt1 = (0, y + int(offset[1]))
					pt2 = (1000, y + int(offset[1]))
					cv2.line(img, pt1, pt2, (50,50,50), 1)
			
			# Draw trails (past frames)
			for j in range(i):
				trail = (state[j]*scale + offset).astype(int)
				for idx, pos in enumerate(trail):
					color = colors[idx % len(colors)]
					cv2.circle(img, tuple(pos), 1, color, -1)
			
			# Draw current positions and acceleration vectors
			for idx, pos in enumerate(curr_pos):
				pos_scaled = (pos*scale + offset).astype(int)
				color = colors[idx % len(colors)]
				cv2.circle(img, tuple(pos_scaled), 5, color, -1)
				# Scale acceleration vector for visibility if needed
				accel_scaled = (curr_acc[idx]*scale).astype(int)
				end_pt = pos_scaled + accel_scaled
				cv2.line(img, tuple(pos_scaled), tuple(end_pt), color, 1)
			
			cv2.imshow('Particles Sim', img)
			key = cv2.waitKey(1)
			if key == ord('r'):
				break
			elif key == ord('c'):
				continue
			elif key != -1:
				return
