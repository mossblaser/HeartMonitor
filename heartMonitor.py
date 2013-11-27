#!/usr/bin/python

import cv
import numpy
import pylab
import time

class Camera(object):
	
	def __init__(self, camera = 0):
		"""
		A simple web-cam wrapper.
		"""
		self.cam = cv.CaptureFromCAM(camera)
		
		if not self.cam:
			raise Exception("Camera not accessible.")
	
	
	def get_frame(self):
		"""
		Return the most recent (successful) image from the webcam
		"""
		frame = None
		while not frame:
			frame = cv.QueryFrame(self.cam)
		
		return frame
	
	
	def get_fps(self):
		fps = cv.GetCaptureProperty(self.cam, cv.CV_CAP_PROP_FPS)
		return fps if fps != -1 else 30.0
	
	
	def get_size(self):
		w = int(cv.GetCaptureProperty(self.cam, cv.CV_CAP_PROP_FRAME_WIDTH))
		h = int(cv.GetCaptureProperty(self.cam, cv.CV_CAP_PROP_FRAME_HEIGHT))
		return (w,h)


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
		Perform an Fast-Fourier-Transform on the buffer and return (magnitude,
		phase) tuples for each of the bins.
		"""
		# Get the "ideal" evenly spaced times
		even_times = numpy.linspace(self.buf[0][0], self.buf[-1][0], len(self.buf))
		
		# Interpolate the data to generate evenly temporally spaced samples
		interpolated = numpy.interp(even_times, *zip(*self.buf))
		
		# Perform the FFT
		fft = numpy.fft.rfft(interpolated)
		return zip(numpy.abs(fft), numpy.angle(fft))
	
	
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
		Get the current beats-per-minute, the phase and the band of FFT data within
		the allowed heart-rate range as a list of (bpm, (magnitude,phase)) tuples.
		"""
		
		fft = self.get_fft()
		
		# Get the bin numbers of the bounds of the possible allowed heart-rates in
		# the FFT
		min_bin = self.bpm_to_bin(self.min_bpm)
		max_bin = self.bpm_to_bin(self.max_bpm)
		
		# Find the bin with the highest intensity (the heartbeat)
		if min_bin == max_bin:
			best_bin = min_bin
		else:
			best_bin = max(range(min_bin, max_bin),
			               key=(lambda i: fft[i-1][0]))
		heartrate = self.bin_to_bpm(best_bin)
		phase     = fft[best_bin-1][1]
		
		# Produce the FFT data in the format described above
		fft_data = zip((self.bin_to_bpm(b) for b in range(min_bin, max_bin+1)),
		               fft[min_bin-1:max_bin])
		
		return heartrate, phase, fft_data
	
	
	@property
	def buf_full(self):
		return len(self.buf) >= self.buf_size
	
	
	@property
	def ready(self):
		return len(self.buf) >= 2
	
	
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


class FaceTracker(object):
	
	def __init__(self, frame, face_position,
	             fh_x = 0.5,  fh_y = 0.13,
	             fh_w = 0.25, fh_h = 0.15):
		"""
		A motion tracker that can track a face (and forehead). Note: This class
		simply provides the interface but doesn't actually track the face as it
		moves.
		@param frame         The first frame containing the face
		@param face_position The position of the face in the frame
		@param fh_x          The x-position on the face of the center of the forehead
		@param fh_y          The y-position on the face of the center of the forehead
		@param fh_w          The width, relative to the face, of the forehead
		@param fh_h          The height, relative to the face, of the forehead
		"""
		
		self.face_position = face_position
		
		self.fh_x = fh_x
		self.fh_y = fh_y
		self.fh_w = fh_w
		self.fh_h = fh_h
	
	
	def update(self, time, frame, face_position = None):
		"""
		Add a new frame. Will override the face position if specified.
		"""
		self.face_position = face_position or self.face_position
	
	
	def get_face(self):
		return self.face_position
	
	
	def get_forehead(self):
		"""
		Get the position of the forehead as tracked by the MotionTracker
		"""
		x,y,w,h = self.get_face()
		
		x += w * self.fh_x
		y += h * self.fh_y
		w *= self.fh_w
		h *= self.fh_h
		
		x -= (w / 2.0)
		y -= (h / 2.0)
		
		return tuple(map(int, (x,y,w,h)))


class Annotator(object):
	
	THICK  = 3 # Thick line width
	THIN   = 1 # Thin line width
	BORDER = 2 # Additional width for outlines
	
	# Colour (Fill, Outline)
	COLOUR_OK   = ((0,255,0), (0,0,0))
	COLOUR_BUSY = ((0,0,255), (0,0,0))
	
	COLOUR_FACE     = (0,255,255)
	COLOUR_FOREHEAD = (0,255,0)
	
	PULSE_SIZE = (9,12)        # Size of the pluse-blob (normal, on pulse)
	PULSE_PHASE = numpy.pi / 4 # Phase during which pulse occurs
	
	SMALL_PULSE_SIZE = 6 # Size of the small pluse-blob
	
	HEAD_WIDTH_SCALE = 0.8 # Scale head width for appearence's sake
	
	FFT_HEIGHT = 0.4 # Height of the FFT on the image
	
	
	def __init__(self):
		"""
		Can annotate various features onto frames.
		"""
		# Setup fonts
		self.large_font = self._get_font(1,Annotator.THICK)
		self.large_font_outline = self._get_font(1,Annotator.THICK + Annotator.BORDER)
		
		self.small_font = self._get_font(0.5,Annotator.THIN)
		self.small_font_outline = self._get_font(0.5,Annotator.THIN + Annotator.BORDER)
		
		# Text colour
		self.colour = Annotator.COLOUR_BUSY
		
		self.forehead = (0,0,1,1)
		self.face     = (0,0,1,1)
	
	
	def _get_font(self, size=1, weight=1, italic=0):
		return cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX,
		                   size, size, italic, weight) 
	
	
	def set_busy(self, busy):
		self.colour = Annotator.COLOUR_OK if not busy else Annotator.COLOUR_BUSY
	
	
	def set_forehead(self, forehead):
		self.forehead = tuple(map(int, forehead))
	
	def set_face(self, face):
		self.face = tuple(map(int, face))
	
	@property
	def metrics(self):
		x,_,w,h = map(int, self.forehead)
		_,y,_,_ = map(int, self.face)
		return (x,y,w,h)
	
	
	def get_colour(self):
		return self.colour
	
	
	def draw_bpm(self, frame, bpm):
		x,y,w,h = self.metrics
		c = self.get_colour()
		
		cv.PutText(frame, "%0.0f"%bpm, (x,y), self.large_font_outline, c[1])
		cv.PutText(frame, "%0.0f"%bpm, (x,y), self.large_font, c[0])
	
	
	def draw_phase(self, frame, phase):
		x,y,w,h = self.metrics
		c = self.get_colour()
		
		x -= int(Annotator.PULSE_SIZE[1] * 1.5)
		y -= Annotator.PULSE_SIZE[1]
		
		if (phase % (2.0 * numpy.pi)) < Annotator.PULSE_PHASE:
			radius = Annotator.PULSE_SIZE[1]
		else:
			radius = Annotator.PULSE_SIZE[0]
		
		cv.Circle(frame, (x,y), radius + Annotator.BORDER, c[1], -1)
		cv.Circle(frame, (x,y), radius, c[0], -1)
	
	
	def draw_face(self, frame):
		x,y,w,h = self.face
		
		# Center of the face
		x += w/2
		y += h/2
		
		# Slightly narrow the elipse to fit most faces better
		w *= Annotator.HEAD_WIDTH_SCALE
		
		c = Annotator.COLOUR_FACE
		
		cv.Ellipse(frame, (int(x),int(y)), (int(w/2),int(h/2)), 0, 0, 360, c, Annotator.THIN)
	
	
	def draw_forehead(self, frame):
		x,y,w,h = self.forehead
		c = Annotator.COLOUR_FOREHEAD
		
		cv.Rectangle(frame, (int(x),int(y)), (int(x+w),int(y+h)), c, Annotator.THIN)
	
	
	def draw_fft(self, frame, fft_data, min_bpm, max_bpm):
		w = frame.width
		h = int(frame.height * Annotator.FFT_HEIGHT)
		x = 0
		y = frame.height
		
		max_magnitude = max(d[1][0] for d in fft_data)
		
		def get_position(i):
			point_x = int(w * (float(fft_data[i][0] - min_bpm) / float(max_bpm - min_bpm)))
			point_y = int(y - ((h * fft_data[i][1][0]) / max_magnitude))
			return point_x, point_y
		
		line = [get_position(i) for i in range(len(fft_data))]
		
		cv.PolyLine(frame, [line], False, self.get_colour()[0], 3)
		
		# Label the largest bin
		max_bin = max(range(len(fft_data)), key=(lambda i: fft_data[i][1][0]))
		
		x,y = get_position(max_bin)
		c = self.get_colour()
		text = "%0.1f"%fft_data[max_bin][0]
		
		cv.PutText(frame, text, (x,y), self.small_font_outline, c[1])
		cv.PutText(frame, text, (x,y), self.small_font, c[0])
		
		# Pulse ring
		r = Annotator.SMALL_PULSE_SIZE
		phase = int(((fft_data[max_bin][1][1] % (2*numpy.pi)) / numpy.pi) * 180)
		cv.Ellipse(frame, (int(x-(r*1.5)),int(y-r)), (int(r),int(r)), 0, 90, 90-phase, c[1], Annotator.THIN+Annotator.BORDER)
		cv.Ellipse(frame, (int(x-(r*1.5)),int(y-r)), (int(r),int(r)), 0, 90, 90-phase, c[0], Annotator.THIN)
		


class Program(object):
	
	def __init__(self,
	             webcam = 0,
	             sample_duration = 10,
	             window_title = "Heart Monitor"):
		"""
		Program to monitor heartrates using a webcam.
		"""
		
		self.cam           = Camera(webcam)
		self.face_detector = FaceDetector(*self.cam.get_size())
		self.face_tracker  = None
		self.heart_monitor = HeartMonitor(sample_duration, fps = self.cam.get_fps())
		self.annotator     = Annotator()
		self.window        = window_title
		
		cv.NamedWindow(self.window)
		
		self.show_bpm      = True
		self.show_face     = True
		self.show_forehead = True
		self.show_fft      = True
	
	
	def find_face(self, frame):
		# Try and find a face
		face = self.face_detector.get_best_face(frame)
		
		if face is not None:
			# Track the new face
			self.face_tracker = FaceTracker(frame, face[0])
	
	
	def sample_frame(self, frame):
		# Get an average of the green channel in on the forehead
		cv.SetImageROI(frame, self.face_tracker.get_forehead())
		sample = cv.Avg(frame)[1]
		cv.ResetImageROI(frame)
		
		return sample
	
	
	def update(self):
		"""
		Mainloop body. Returns True unless termination requested.
		"""
		
		frame = self.cam.get_frame()
		frame_time = time.time()
		
		if self.face_tracker is None:
			# No face known
			self.find_face(frame)
		else:
			# Track the face
			self.face_tracker.update(frame_time, frame)
			self.annotator.set_face(self.face_tracker.get_face())
			self.annotator.set_forehead(self.face_tracker.get_forehead())
			
			# Update the heart monitor
			self.heart_monitor.add_sample(frame_time, self.sample_frame(frame))
			self.annotator.set_busy(not self.heart_monitor.buf_full)
			
			if self.heart_monitor.ready:
				bpm, phase, fft_data = self.heart_monitor.get_bpm()
				
				# Draw the OSD
				if fft_data and self.show_fft:
					self.annotator.draw_fft(frame, fft_data,
					                        self.heart_monitor.min_bpm,
					                        self.heart_monitor.max_bpm)
				
				if self.show_face:
					self.annotator.draw_face(frame)
				
				if self.show_forehead:
					self.annotator.draw_forehead(frame)
				
				if self.show_bpm:
					self.annotator.draw_bpm(frame, bpm)
					self.annotator.draw_phase(frame, phase)
		
		# Display the (possibly annotated) frame
		cv.ShowImage(self.window, frame)
		
		# Handle keypresses
		key = cv.WaitKey(10) & 255
		if key == 27: # Escape
			# Exit
			return False
		elif key == ord("r"):
			# Reset the heart monitor and face tracker
			self.face_tracker = None
			self.heart_monitor.reset()
		elif key == ord(" "):
			# Re-find the face
			self.face_tracker = None
		elif key == ord("1"):
			self.show_face = not self.show_face
		elif key == ord("2"):
			self.show_forehead = not self.show_forehead
		elif key == ord("3"):
			self.show_fft = not self.show_fft
		elif key == ord("4"):
			self.show_bpm = not self.show_bpm
		
		return True
	
	
	def run(self):
		"""
		Blocks running the mainloop
		"""
		
		while self.update():
			pass


if __name__=="__main__":
	Program().run()

