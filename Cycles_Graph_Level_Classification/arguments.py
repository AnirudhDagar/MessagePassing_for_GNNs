import configargparse

def buildParser():
	parser = configargparse.ArgParser()

	# Data setup
	parser.add('--datapath', default='resources/CYCLE/', help='Data Path Destination')
	parser.add('--savepath', default='resources/saved/', help='Saved Path Destination')
	parser.add('-w', '--weight_filename', default='saved_weights_cycle', help='Saved weights Destination')
	parser.add('-p', '--predictions_filename', default='py_predicted_cycle', help='Output predictions for comparision')

	# Training setup
	parser.add('-s', '--seed', help='Seed for random number generation', type=int, default=2020)
	parser.add('-e', '--epochs', help='Number of epochs', type=int, default=400)
	parser.add('--no_vis', dest='no_vis', action='store_true', help='Do not create Visualization')
	parser.add('--print_freq', help='Frequency of printing updates between epochs', type=int, default=20)

	# Optimizer setup
	parser.add('-l', '--lr', help='Learning rate', type=float, default=0.01)
	parser.add('-m', '--momentum', help='Momentum of optimizer', type=float, default=0.9)

	# Model setup
	parser.add('--nhid1', help='hidden dimension 1', type=int, default=6)
	parser.add('--nhid2', help='hidden dimension 2', type=int, default=4)
	parser.add('--out', help='out dimension', type=int, default=2)

	return parser
