import torch
import torch.nn as nn 

class RNN(nn.Module):

	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.output_size = output_size

		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		combined = torch.cat((input, hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		output = self.softmax(output)
		return output, hidden 
	
	def init_hidden(self, device='cpu'):
		if device == 'cpu':
			return torch.zeros(1, self.hidden_size)
		else:
			return torch.zeros(1, self.hidden_size).to(device)
		#return torch.zeros(1, self.hidden_size) 
