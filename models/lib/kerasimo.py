from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import Model
import numpy as np
import math
import collections

maxheight = 0

class Neuron:
	def __init__(self, x, y, a):
		self.x = x
		self.y = y
		self.a = a

class DenseLayer:
	def __init__(self, layer, shape, columns, activity):
		global maxheight
		self.layer = layer
		self.shape = shape
		self.activity = activity
		self.columns = columns
		self.n = len(activity)
		maxheight = max(maxheight, self.GetHeight())

	def GetWidth(self):
		return self.columns*25

	def GetHeight(self):
		return (self.n//self.columns)*25

	def GetCoordinates(self):
		points = list()
		for i in range(0, self.n):
			points.append(Neuron(
				(i % self.columns)*25,
				(i // self.columns)*25,
				self.activity[i]))
		return points


class ConvolutedLayer:
	def __init__(self, layer, shape, columns, activity):
		global maxheight
		self.nx = shape[1]
		self.ny = shape[2]
		self.nz = shape[3]
		self.layer = layer
		self.shape = shape
		self.activity = activity
		self.columns = columns
		self.n = len(activity)
		maxyn = self.ny*self.nz + 2*self.nz
		maxheight = max(maxheight, self.GetHeight())

	def GetWidth(self):
		return self.nx*self.columns*25 + self.columns * 50

	def GetHeight(self):
		rows = self.nz // self.columns
		return self.ny*25*rows + rows*50

	def GetCoordinates(self):
		points = list()

		for ky in range(0, self.nz // self.columns):
			for kx in range(0, self.columns):
				maxa = 0
				for j in range(0, self.ny):
					for i in range(0, self.nx):
						maxa = max(maxa, self.activity[i][j][kx+ky*self.columns])
				for j in range(0, self.ny):
					for i in range(0, self.nx):
						points.append(Neuron(
							i * 25 + self.nx*kx*25 + kx*50,
							j * 25 + ky*self.ny*25 + ky*50,
							self.activity[i][j][kx+ky*self.columns]/(maxa*0.5)
						))
		return points

def AddLine(strlist, p1, p2):
	dx = p2.x - p1.x
	dy = p2.y - p1.y
	r = math.sqrt(dx*dx + dy*dy)
	dx = dx / r
	dy = dy / r
	strlist.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#888" stroke-width="1" marker-end="url(#arrow)" />\n'
	% (p2.x-dx*10, p2.y-dy*10, p1.x+dx*18, p1.y+dy*18))

def AddCircle(strlist, p):
	colorr = 0
	colorb = 0
	if (p.a>0):
		colorr = int(min(p.a, 1.)*255)
	else:
		colorb = int(min(-p.a, 1.)*255);
	strlist.append('<circle cx="%d" cy="%d" r="10" stroke="black" stroke-width="1" fill="rgb(%d,0,%d)" />\n'
	% (p.x, p.y, colorr, colorb))

def CalcNeuronCoordinates(layeridx, layers):
	global maxheight
	width = 70
	points = layers[layeridx].GetCoordinates()
	x1 = 10
	for i in range(0, layeridx): x1 = x1 + width + layers[i].GetWidth()
	y1 = 10 + (maxheight-layers[layeridx].GetHeight()) / 2.
	for p in points:
		p.x = p.x + x1
		p.y = p.y + y1
	return points

def GetSize(layers):
	width = 20
	height = 20
	for l in layers:
		width = width + l.GetWidth() + 70
		height = max(height, l.GetHeight())
	return (width, height)

def WriteSVG(f, layers):
	global maxheight
	xrect = 0
	layeridx = 0
	circlelist = list()
	linelist = list()
	for l in layers:
		neurons1 = CalcNeuronCoordinates(layeridx, layers)
		for n in neurons1: AddCircle(circlelist, n)
		if (layeridx != 0):
			neurons2 = CalcNeuronCoordinates(layeridx-1, layers)
			for n1 in neurons1:
				for n2 in neurons2:
					AddLine(linelist, n1, n2)
		circlelist.append("\n")
		linelist.append("\n")
		#--------
		#rectcolor = 220
		#if (layeridx&1) == 0: rectcolor = 255
		#f.write('<rect x="%d" y="%d" width="%d" height="%d" fill="rgb(%d,%d,%d)"/>\n'
		#	% (xrect, 0, l.GetWidth()+70, maxheight, rectcolor, rectcolor, rectcolor))
		#xrect = xrect + l.GetWidth() + 70
		#-------
		layeridx = layeridx + 1;
	for lstr in linelist: f.write(lstr)
	for cstr in circlelist: f.write(cstr)


def ToSVG(name, model, X, **kwargs):
	columns = kwargs.get('columns', [1 for i in range(len(model.layers)+1)])

	for m in model.layers:
		print("====================================================================")
		print(m.get_config())
		if m.get_weights():
			print('weights list len: ', len(m.get_weights()))
			for w in  m.get_weights():
				print('weights shape: ', w.shape, ' total: ',  w.size)
		print('input shape:  ', m.input_shape)
		print('output shape: ', m.output_shape)
	print("====================================================================")

	samples = list()
	for x in X:
		if model.layers[0].get_config()['name'].startswith('dense'):
			samples.append(list([DenseLayer(model.layers[0], model.layers[0].input_shape, columns[0], x)]))
		if model.layers[0].get_config()['name'].startswith('conv2d'):
			samples.append(list([ConvolutedLayer(model.layers[0], model.layers[0].input_shape, columns[0], x)]))
		if model.layers[0].get_config()['name'].startswith('zero_padding2d'):
			samples.append(list([ConvolutedLayer(model.layers[0], model.layers[0].input_shape, columns[0], x)]))
		if model.layers[0].get_config()['name'].startswith('max_pooling2d'):
			samples.append(list([ConvolutedLayer(model.layers[0], model.layers[0].input_shape, columns[0], x)]))

	i = 1
	for l in model.layers:
		intermediate_model = Model(inputs=model.input, outputs=l.output)
		result = intermediate_model.predict(X)
		for j in range(0, len(result)):
			if l.get_config()['name'].startswith('dense'):
				samples[j].append(DenseLayer(l, l.output_shape, columns[i], result[j]))
			if l.get_config()['name'].startswith('conv2d'):
				samples[j].append(ConvolutedLayer(l, l.output_shape, columns[i], result[j]))
			#if l.get_config()['name'].startswith('zero_padding2d'):
			#	samples[j].append(ConvolutedLayer(l, l.output_shape, columns[i], result[j]))
			if l.get_config()['name'].startswith('max_pooling2d'):
				samples[j].append(ConvolutedLayer(l, l.output_shape, columns[i], result[j]))
		i = i + 1

	print('Plotted layers + input: %d' % len(samples[0]))
	(width, height) = GetSize(samples[0])
	print('width: %d, height: %d' % (width, height))

	for i in range(0, len(samples)):
		filename = '%s%02d.svg' % (name, i)
		print('Store file %s' % filename)
		f = open(filename, 'w')
		f.write('<?xml version="1.0" encoding="UTF-8"?>\n');
		f.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" baseProfile="full" width="%dpx" height="%dpx">\n' % (width, height));
		f.write('<defs>\n')
		f.write('<marker id="arrow" markerWidth="10" markerHeight="10" refX="0" refY="3" orient="auto" markerUnits="strokeWidth">\n')
		f.write('<path d="M0,0 L0,6 L9,3 z" fill="#888" />\n')
		f.write('</marker>\n')
		f.write('</defs>\n');
		WriteSVG(f, samples[i])
		f.write("</svg>\n");
		f.close()
