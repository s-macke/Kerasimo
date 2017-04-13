# Library Kerasimo
# you might have to set "export LANG=en_US.UTF-8"

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
	def __init__(self, layer, columns, activity):
		global maxheight
		self.layer = layer
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
		maxa = -1e99
		mina =  1e99
		for i in range(0, self.n):
			a = self.activity[i]
			maxa = max(maxa, a)
			mina = min(mina, a)
		for i in range(0, self.n):
			a = self.activity[i]
			if self.layer and self.layer.get_config() and 'activation' in self.layer.get_config():
				if self.layer.get_config()['activation'] == 'relu':
					if (maxa != 0): a = a/(maxa*0.5)
			points.append(Neuron(
				(i % self.columns)*25,
				(i // self.columns)*25,
				a))
		return points


class ConvolutedLayer:
	def __init__(self, layer, columns, activity):
		global maxheight
		self.layer = layer
		self.activity = activity
		#self.activity = np.transpose(activity, (1,2,0))
		self.nx = self.activity.shape[0]
		self.ny = self.activity.shape[1]
		self.nz = self.activity.shape[2]
		self.columns = columns
		self.n = len(self.activity)
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
				maxa = -1e99
				mina =  1e99
				for j in range(0, self.ny):
					for i in range(0, self.nx):
						a = self.activity[i][j][kx+ky*self.columns]
						maxa = max(maxa, a)
						mina = min(mina, a)
				for j in range(0, self.ny):
					for i in range(0, self.nx):
						a = self.activity[i][j][kx+ky*self.columns]
						if self.layer and self.layer.get_config() and 'activation' in self.layer.get_config():
							if self.layer.get_config()['activation'] == 'relu':
								if (maxa != 0): a = a/(maxa*0.5)
						points.append(Neuron(
							i * 25 + self.nx*kx*25 + kx*50,
							j * 25 + ky*self.ny*25 + ky*50,
							a
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

def WriteSVG(f, layers, showarrows):
	global maxheight
	xrect = 0
	layeridx = 0
	circlelist = list()
	linelist = list()
	for l in layers:
		neurons1 = CalcNeuronCoordinates(layeridx, layers)
		for n in neurons1: AddCircle(circlelist, n)
		if (layeridx != 0) and (showarrows):
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
	showarrows   = kwargs.get('showarrows', True)
	batch_size   = kwargs.get('batch_size', 32)
	showreshape = kwargs.get('showreshape', False)

	print('Kerasimo')
	print('  class:         ', model.__class__.__name__);
	print('  layers:        ', len(model.layers));
	print('  columns:        ', columns);
	print('  training data: ', X.shape);

	for m in model.layers:
		print("====================================================================")
		print(m.__class__.__name__)
		#if (m.__class__.__name__ == 'Lambda'): continue
		if "get_config" in dir(m):
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
		if model.layers[0].__class__.__name__ == 'InputLayer':
			samples.append(list([ConvolutedLayer(model.layers[0], columns[0], x)]))
		if model.layers[0].__class__.__name__ == 'Dense':
			samples.append(list([DenseLayer(model.layers[0], columns[0], x)]))
		if model.layers[0].__class__.__name__ == 'Conv2D':
			samples.append(list([ConvolutedLayer(model.layers[0], columns[0], x)]))
		if model.layers[0].__class__.__name__ == 'ZeroPadding2D':
			samples.append(list([ConvolutedLayer(model.layers[0], columns[0], x)]))
		if model.layers[0].__class__.__name__ == 'MaxPooling2D':
			samples.append(list([ConvolutedLayer(model.layers[0], columns[0], x)]))

	print('generated list for ', len(samples), ' samples')
	if (len(samples) == 0): return

	i = 1
	for l in model.layers:
		intermediate_model = Model(inputs=model.input, outputs=l.output)
		result = intermediate_model.predict(X, batch_size=batch_size)
		print('train to layer: ', i, ' with result len: ', result.shape)
		for j in range(0, len(result)):
			if l.__class__.__name__ == 'Dense':
				samples[j].append(DenseLayer(l, columns[i], result[j]))
			if l.__class__.__name__ == 'Flatten' and showreshape:
				samples[j].append(DenseLayer(l, columns[i], result[j]))
			if l.__class__.__name__ == 'Conv2D':
				samples[j].append(ConvolutedLayer(l, columns[i], result[j]))
			if l.__class__.__name__ == 'Reshape' and showreshape:
				samples[j].append(ConvolutedLayer(l, columns[i], result[j]))
			if l.__class__.__name__ == 'Conv2DTranspose':
				samples[j].append(ConvolutedLayer(l, columns[i], result[j]))
			#if l.__class__.__name__ == 'ZeroPadding2D':
			#	samples[j].append(ConvolutedLayer(l, l.output_shape, columns[i], result[j]))
			if l.__class__.__name__ == 'MaxPooling2D':
				samples[j].append(ConvolutedLayer(l, columns[i], result[j]))
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
		WriteSVG(f, samples[i], showarrows)
		f.write("</svg>\n");
		f.close()
