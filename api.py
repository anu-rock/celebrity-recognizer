# Copyright 2018 Anurag Bhandari and AI Creatives Meetup Group. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==============================================================================

import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES

# App configuration
app = Flask(__name__)
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
configure_uploads(app, photos)

# Global variables
model_file = "trained-model/output_graph.pb"
label_file = "trained-model/output_labels.txt"
input_layer = "Placeholder"
output_layer = "final_result"

@app.route('/infer', methods=['POST'])
def infer():
	"""
	Performs inference on the given target image,
	and returns the response as JSON.
	"""
	if 'target' in request.files:
		file_name = photos.save(request.files['target'])
		graph = load_graph(model_file)
		t = read_tensor_from_image_file("uploads/" + file_name)
		input_name = "import/" + input_layer
		output_name = "import/" + output_layer
		input_operation = graph.get_operation_by_name(input_name)
		output_operation = graph.get_operation_by_name(output_name)

		with tf.Session(graph=graph) as sess:
			results = sess.run(output_operation.outputs[0], {
				input_operation.outputs[0]: t
			})
		results = np.squeeze(results)

		top_k = results.argsort()[-5:][::-1]
		labels = load_labels(label_file)
		response = []
		for i in top_k:
			response.append({"name": labels[i], "confidence": str(results[i])})
		return jsonify(response)
	else:
		return jsonify({"error": "No target specified."})

def load_graph(model_file):
	"""
	Loads the specified graph (model) in memory for further processing.

	Parameters
	----------
	model_file : string
		Absolute or relative path of the graph file (eg. output_graph.pb)
	"""
	graph = tf.Graph()
	graph_def = tf.GraphDef()

	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)

	return graph

def load_labels(label_file):
	"""
	Reads graph labels into memory from file.

	Parameters
	----------
	label_file : string
		Absolute or relative path of the labels file (eg. output_labels.txt)
	"""
	label = []
	proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
	for l in proto_as_ascii_lines:
		label.append(l.rstrip())
	return label

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
								input_mean=0, input_std=255):
	"""
	Reads the given image file as a tensor.

	Parameters
	----------
	file_name : string
		Absolute or relative path of the image file (eg. bill.jpg)
	"""
	input_name = "file_reader"
	output_name = "normalized"
	file_reader = tf.read_file(file_name, input_name)
	if file_name.endswith(".png"):
		image_reader = tf.image.decode_png(
			file_reader, channels=3, name="png_reader")
	elif file_name.endswith(".gif"):
		image_reader = tf.squeeze(
			tf.image.decode_gif(file_reader, name="gif_reader"))
	elif file_name.endswith(".bmp"):
		image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
	else:
		image_reader = tf.image.decode_jpeg(
			file_reader, channels=3, name="jpeg_reader")
	float_caster = tf.cast(image_reader, tf.float32)
	dims_expander = tf.expand_dims(float_caster, 0)
	resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
	normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
	sess = tf.Session()
	result = sess.run(normalized)

	return result

if __name__ == "__main__":
	app.run()
