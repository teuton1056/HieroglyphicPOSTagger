import data_setup 
import sys 
import torch 
import random 
import tqdm

#rnn = torch.load('model.pt')
#device = 'cpu'

def random_test_example():
	pos = random.choice(data_setup.pos_list) 
	random.shuffle(data_setup.test_pos_dict[pos])
	if len(data_setup.test_pos_dict[pos]) > 1:
		word = data_setup.test_pos_dict[pos].pop(0)
	else:
		word = data_setup.test_pos_dict[pos][0]
		data_setup.test_pos_dict.pop(pos)
	return pos, word

def evaluate(word_tensor, model='model.pt'):
	rnn = torch.load(model)
	hidden = rnn.init_hidden(device='cuda:0')
	for i in range(word_tensor.size()[0]):
		output, hidden = rnn(word_tensor[i], hidden)
	return output 

def predict(word, model='model.pt'):
	word_tensor = data_setup.wordToTensor(word).cuda()
	output = evaluate(word_tensor, model)
	n = len(data_setup.pos_list)
	topv, topi = output.data.topk(n, 1, True)
	predictions = [] 

	for i in range(n):
		value = topv[0][i] 
		pos_index = topi[0][i]
		predictions.append((value, data_setup.pos_list[pos_index]))
	return predictions 

def predict_cli(word, model='model.pt'):
	print("Predicting: {}".format(word))
	predictions = predict(word, model)
	for prediction in predictions:
		print('({})\t- {}'.format(round(float(prediction[0]),5), prediction[1]))

def predict_html(word, full=False, model='model.pt'):
	predictions = predict(word, model)
	html = '<span class="word">{}</span>'.format(word)
	if full:
		html += '<table> <tr>'
		html += '<th>Part of Speech</th> <th>Score</th>'
		for prediction in predictions:
			html += f"<tr> <td>{prediction[1]}</td> <td>{round(float(prediction[0]),5)}</td> </tr>"
			#html += '<div class="prediction">{}</div>'.format(prediction[1])
		html += '</tr> </table>'
		return html
	else:
		html += f'<br><span>Prediction: {predictions[0][1]} </span>'
	return html

def is_correct(pos, _predictions, grade=1):
	predictions = top_n_predictions(_predictions, 1)
	return pos in [p[1] for p in predictions]

def top_n_predictions(predictions, n):
	return sorted(predictions, key=lambda x: x[0], reverse=True)[:n]

def log(text):
	with open('test_log.txt', 'a', encoding='utf-8') as f:
		f.write(text + '\n')

def evaluate_model(tests=1000, model='model.pt'):
	correct = 0 
	total = 0 
	for i in tqdm.tqdm(range(tests)):
		#print(word)
		pos, word = random_test_example()
		predictions = predict(word, model)
		if is_correct(pos, predictions, 1):
			correct += 1 
			log('Correct: {}, {}'.format(word, pos))
		else:
			log('Incorrect: {}, {}'.format(word, pos))
		total += 1 
	print('Accuracy: {}%'.format(correct / total * 100))

if __name__ == "__main__":
	#predict_cli("𓇋𓅱")
	evaluate_model(5000, 'model_1.pt')