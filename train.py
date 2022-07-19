import torch 
import data_setup 
from model import RNN 
import random 
import time 
import math  

n_hidden = 1024 
n_epochs = 200000
print_every = 1000
plot_every = 1000
learning_rate = 0.0007

if n_epochs < len(data_setup.pos_list):
	print('Epochs is set to a larger number than the number of words in the dataset.')
	print('Setting epochs to the number of words in the dataset.')
	n_epochs = len(data_setup.pos_list)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))
if torch.cuda.is_available(): print("Device is {}".format(torch.cuda.get_device_name(0)))

def pos_from_output(output):
	# converts an output to a pos
	n, i = output.data.topk(1) 
	pos_i = i[0].item()
	return data_setup.pos_list[pos_i], pos_i 

def random_choice(L):
	# chooses a random element from a list
	return L[random.randint(0, len(L) - 1)]

def random_training_example():
	# chooses a random word from the word list
	pos = random_choice(data_setup.pos_list) 
	word = random_choice(data_setup.pos_dict[pos])
	pos_tensor = torch.tensor([data_setup.pos_list.index(pos)]).cuda()
	word_tensor = data_setup.wordToTensor(word).cuda()
	return pos, word, pos_tensor, word_tensor 

def criterion(output, target):
	# calculates the loss
	return torch.nn.NLLLoss()(output, target)
	return torch.nn.functional.cross_entropy(output, target)

rnn = RNN(len(data_setup.alphabet), n_hidden, len(data_setup.pos_list))
rnn.to(device)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(pos_tensor, word_tensor):

	hidden = rnn.init_hidden(device)
	rnn.zero_grad()
	for i in range(word_tensor.size()[0]):
		output, hidden = rnn(word_tensor[i], hidden)
	loss = criterion(output, pos_tensor)
	loss.backward() 

	optimizer.step()

	return output, loss.item()

# main 
if __name__ == "__main__":
	current_loss = 0 
	all_losses = [] 

	def timeSince(since):
		now = time.time()
		s = now - since
		m = math.floor(s / 60)
		s -= m * 60
		return '%dm %ds' % (m, s)
	
	start = time.time() 

	for epoch in range(1, n_epochs + 1):
		pos, word, pos_tensor, word_tensor = random_training_example()
		output, loss = train(pos_tensor, word_tensor)
		current_loss += loss 
		
		if epoch % print_every == 0:
			guess, guess_i = pos_from_output(output)
			correct = '✓' if guess == pos else '✗ (%s)' % pos
			print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, word, guess, correct))

		if epoch % plot_every == 0:
			all_losses.append(current_loss / plot_every)
			current_loss = 0
	
	torch.save(rnn, 'model_1.pt')

