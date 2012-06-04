#!/usr/bin/python

import cv
import numpy
import pylab
import time
from colorsys import rgb_to_hsv

class Camera(object):
	
	def __init__(self, camera = 0):
		self.cam = cv.CaptureFromCAM(camera)
	
	def get_frame(self):
		return cv.QueryFrame(self.cam)


class FaceDetector(object):
	
	def __init__(self, width, height, cascade_file="haarcascade_frontalface_alt.xml"):
		self.cascade = cv.Load(cascade_file)
		self.storage = cv.CreateMemStorage()
		
		self.gray = cv.CreateImage((width, height), 8, 1)
	
	def get_faces(self, image):
		cv.CvtColor(image, self.gray, cv.CV_BGR2GRAY)
		cv.EqualizeHist(self.gray, self.gray)
		
		return cv.HaarDetectObjects(self.gray, self.cascade, self.storage,
		                            scale_factor = 1.3,
		                            min_neighbors = 2,
		                            flags = cv.CV_HAAR_DO_CANNY_PRUNING,
		                            min_size = (40,40))
	
	def get_best_face(self, image):
		faces = self.get_faces(image)
		best = None
		for face in faces:
			if best is None or face[1] > best[1]:
				best = face
		
		return best


class HeartMonitor(object):
	
	def __init__(self, window_size, fps = 30, min_bpm = 50, max_bpm = 200):
		self.num_samples = int(window_size*fps)
		self.fps = fps
		self.min_bpm = min_bpm
		self.max_bpm = max_bpm
		
		self.time = []
		self.data = []
		
		self.bins = []
	
	
	def get_bpm(self):
		interpolated = numpy.interp(
		                 numpy.linspace(self.time[0], self.time[-1],
		                                len(self.time)),
		                 self.time,
		                 self.data)
		fft = numpy.abs(numpy.fft.rfft(interpolated))
		
		fps = float(len(self.time)) / (self.time[-1] - self.time[0])
		
		# Number of bins in the full FFT
		bins = len(self.data)
		
		def bin_to_bpm(bin):
			return (60.0 * bin * fps) / float(bins)
		# Calculate the BPM of this bin
		
		f = open("fft","w")
		for bin in range(len(fft)):
			f.write("%s %s\n"%(bin_to_bpm(bin+1), fft[bin]))
		
		# Get the index of the min and max bins to scan
		min_bin = int(float(bins * self.min_bpm) / float(60.0 * fps))
		max_bin = int(float(bins * self.max_bpm) / float(60.0 * fps))
		
		# Find the bin with the highest intensity
		best_bins = sorted(range(min_bin, max_bin),
		                   key=(lambda i: fft[i-1]),
		                   reverse=True)
		
		# Store the calucated bins
		self.bins = fft[min_bin-1:max_bin]
		
		return bin_to_bpm(best_bins[0])
	
	
	@property
	def full(self):
		return len(self.data) >= self.num_samples
	
	
	def add(self, time, datum):
		if self.full:
			self.data.pop(0)
			self.time.pop(0)
		self.time.append(time)
		self.data.append(datum)


cam = Camera(0)
cv.NamedWindow("w1")#, cv.CV_WINDOW_AUTO_SIZE)

fd = FaceDetector(640,480)

hm = HeartMonitor(10)

face = None
font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8) 
i = 30
ffts = []
bpm = 0.0
 
while True:
	frame = cam.get_frame()
	if frame:
		while not face:
			face = fd.get_best_face(frame)
			if not face:
				frame = None
				while not frame:
					frame = cam.get_frame()
		x,y,w,h = face[0]
		w /= 2
		h /= 3
		x += w/2
		#y += h/2
		
		w /= 2
		h /= 2
		x += w/2
		y += h/4
		
		cv.SetImageROI(frame, (x,y,w,h))
		hm.add(time.time(), cv.Avg(frame)[1])
		cv.ResetImageROI(frame)
		cv.Rectangle(frame, tuple(map(int, (x,y))), tuple(map(int, (x+w, y+h))), (0,255,0))
		cv.PutText(frame, "%0.1f"%bpm, (x,y), font, (0,255,0))
		
		if ffts:
			colour = (0,255,0) if hm.full else (0,0,255)
			cv.PolyLine(frame, [ffts], False, colour, 3)
		
		if i == 0:
			bpm = hm.get_bpm()
			open("log","a").write(str(bpm) + "\n")
			
			i = 1
			
			if len(hm.bins) > 0:
				ffts = []
				for num,value in enumerate(hm.bins):
					ffts.append((int(640*(float(num)/len(hm.bins))), 
					             240+(240-int(240*(float(value)/max(hm.bins))))))
		i-= 1
		
		cv.ShowImage("w1", frame)
	else:
		print "Failed frame."
	
	key = cv.WaitKey(10) & 255
	if key == 27:
		break
	elif key == ord("r"):
		hm.data = []
		hm.time = []
		ffts = []
		i = 30
	elif key == ord(" "):
		face = None
