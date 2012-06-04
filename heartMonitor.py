#!/usr/bin/python

import cv
import numpy
import pylab
import time
from colorsys import rgb_to_hsv

class Camera(object):
	
	def __init__(self, camera = 0):
		"""
		A simple web-cam wrapper.
		"""
		self.cam = cv.CaptureFromCAM(camera)
	
	def get_frame(self):
		"""
		Return the most recent image from the webcam
		"""
		return cv.QueryFrame(self.cam)


class FaceDetector(object):
	
	def __init__(self, width, height, cascade_file="haarcascade_frontalface_alt.xml"):
		"""
		Detects faces in an image.
		@param width        Width of the images that will be supplied
		@param height       Height of the images that will be supplied
		@param cascade_file Haar cascade data file for fronts of faces
		"""
		
		# Load the cascade
		self.cascade = cv.Load(cascade_file)
		
		# Storage for the algorithm to use
		self.storage = cv.CreateMemStorage()
		
		# A grayscale buffer to copy images for processing into
		self.gray = cv.CreateImage((width, height), 8, 1)
	
	def get_faces(self, image):
		"""
		Given an opencv image, return a ((x,y,w,h), certainty) tuple for each face
		detected.
		"""
		
		# Convert the image to grayscale and normalise
		cv.CvtColor(image, self.gray, cv.CV_BGR2GRAY)
		cv.EqualizeHist(self.gray, self.gray)
		
		# Detect faces
		return cv.HaarDetectObjects(self.gray, self.cascade, self.storage,
		                            scale_factor = 1.3,
		                            min_neighbors = 2,
		                            flags = cv.CV_HAAR_DO_CANNY_PRUNING,
		                            min_size = (40,40))
	
	def get_best_face(self, image):
		"""
		Wrapper around get_faces which returns the face with the highest certainty
		or None if no faces were found
		"""
		try:
			return max(self.get_faces(image),
			           key = (lambda f: f[1]))
		except ValueError:
			return None


class HeartMonitor(object):
	
	def __init__(self, window_duration, fps = 30, min_bpm = 50, max_bpm = 200):
		"""
		Class which detects heart-beats in a sequence of image colour samples.
		@param window_duration The number of seconds of samples to use
		@param fps             The nominal sample rate
		@param min_bpm         Minimum cut-off for possible heartrates
		@param max_bpm         Maximum cut-off for possible heartrates
		"""
		
		self.min_bpm = min_bpm
		self.max_bpm = max_bpm
		
		# The maximum number of samples to buffer
		self.buf_size = int(window_duration*fps)
		
		# Buffer of (timestamp, value) tuples
		self.buf = []
	
	
	@property
	def fps(self):
		"""
		The average framerate/samplerate of the buffer
		"""
		return float(len(self.buf)) / (self.buf[-1][0] - self.buf[0][0])
	
	
	def get_fft(self):
		"""
		Perform an Fast-Fourier-Transform on the buffer and return the magnitude of
		the bins.
		"""
		# Get the "ideal" evenly spaced times
		even_times = numpy.linspace(self.buf[0][0], self.buf[-1][0], len(self.buf))
		
		# Interpolate the data to generate evenly temporally spaced samples
		interpolated = numpy.interp(even_times, *zip(*self.buf))
		
		# Perform the FFT
		return numpy.abs(numpy.fft.rfft(interpolated))
	
	
	def bin_to_bpm(self, bin):
		"""
		Convert an FFT bin number into a heart-rate in beats-per-minute for the
		current framerate. Bin numbers start from 1.
		"""
		
		return (60.0 * bin * self.fps) / float(len(self.buf))
	
	
	def bpm_to_bin(self, bpm):
		"""
		Convert a heart-rate in beats-per-minute into an FFT bin number for the
		current framerate. Bin numbers start from 1.
		"""
		
		return int(float(len(self.buf) * bpm) / float(60.0 * self.fps))
	
	
	def get_bpm(self):
		"""
		Get the current beats-per-minute and the band of FFT data within the allowed
		heart-rate range as a list of (bpm, amplitude) tuples.
		"""
		
		fft = self.get_fft()
		
		# Get the bin numbers of the bounds of the possible allowed heart-rates in
		# the FFT
		min_bin = self.bpm_to_bin(self.min_bpm)
		max_bin = self.bpm_to_bin(self.max_bpm)
		
		# Find the bin with the highest intensity (the heartbeat)
		best_bin = max(range(min_bin, max_bin),
		               key=(lambda i: fft[i-1]))
		heartrate = self.bin_to_bpm(best_bin)
		
		# Produce the FFT data in the format described above
		fft_data = zip((self.bin_to_bpm(b) for b in range(min_bin, max_bin+1)),
		               fft[min_bin-1:max_bin])
		
		return heartrate, fft_data
	
	
	@property
	def buf_full(self):
		return len(self.buf) >= self.buf_size
	
	
	def add_sample(self, time, value):
		"""
		Add a new colour sample
		"""
		if self.buf_full:
			self.buf.pop(0)
		
		self.buf.append((time, value))
	
	
	def reset(self):
		"""
		Reset the heartrate monitor, start from scratch again.
		"""
		self.buf = []


cam = Camera(0)
cv.NamedWindow("w1")#, cv.CV_WINDOW_AUTO_SIZE)

fd = FaceDetector(640,480)

hm = HeartMonitor(10)

face = None
font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8) 
i = 30
ffts = []
bpmh = []
bpm = 0.0

show_face = False
show_bpm = False
show_fft = False

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
		
		if show_face:
			cv.Rectangle(frame, tuple(map(int, (x,y))), tuple(map(int, (x+w, y+h))), (0,255,255))
		
		w /= 2
		h /= 3
		x += w/2
		#y += h/2
		
		w /= 2
		h /= 2
		x += w/2
		y += h/4
		
		cv.SetImageROI(frame, (x,y,w,h))
		hm.add_sample(time.time(), cv.Avg(frame)[1])
		cv.ResetImageROI(frame)
		if show_face:
			cv.Rectangle(frame, tuple(map(int, (x,y))), tuple(map(int, (x+w, y+h))), (0,255,0))
		
		if show_bpm:
			cv.PutText(frame, "%0.0f"%bpm, (x,y), font, (0,255,0))
		
		if ffts:
			if show_fft:
				colour = (0,255,0) if hm.buf_full else (0,0,255)
				cv.PolyLine(frame, [ffts], False, colour, 3)
		
		if i == 0:
			bpm, fft_data = hm.get_bpm()
			bpmh.append(bpm)
			if len(bpmh) > 100:
				bpmh.pop(0)
			bpm = sum(bpmh) / len(bpmh)
			open("log","a").write(str(bpm) + "\n")
			
			i = 1
			
			if len(fft_data) > 0:
				ffts = []
				for num,data in enumerate(fft_data):
					_, value = data
					ffts.append((int(640*(float(num)/len(fft_data))), 
					             240+(240-int(240*(float(value)/max((d[1] for d in fft_data)))))))
		i-= 1
		
		cv.ShowImage("w1", frame)
	else:
		print "Failed frame."
	
	key = cv.WaitKey(10) & 255
	if key == 27:
		break
	elif key == ord("r"):
		hm.reset()
		ffts = []
		i = 30
		face = None
		bpm = 0
	elif key == ord(" "):
		face = None
	elif key == ord("1"):
		show_face = not show_face
	elif key == ord("2"):
		show_fft = not show_fft
	elif key == ord("3"):
		show_bpm = not show_bpm
