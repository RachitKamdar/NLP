import torch
import torchtext
from torchtext.data.utils import get_tokenizer


EMBEDDING_SIZE = 100
HIDDEN_SIZE = 10

device = "cpu"

def train_model(model, loss_function, optimizer, data_loader):
	# set model to training mode
	model.train()
	current_loss = 0.0
	current_acc = 0
	# iterate over the training data
	for i, (inputs, labels) in enumerate(data_loader):
		# zero the parameter gradients
		optimizer.zero_grad()
		with torch.set_grad_enabled(True):
			# forward
			outputs = model(inputs[0])
			#outputs = model(inputs)
			_, predictions = torch.max(outputs, 1)
			loss = loss_function(outputs, labels)
			# backward
			loss.backward()
			optimizer.step()
			# statistics
		current_loss += loss.item() * inputs.size(0)
		current_acc += torch.sum(predictions == labels.data)
	total_loss = current_loss / len(data_loader.dataset)
	total_acc = current_acc.double() / len(data_loader.dataset)
	print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss,total_acc))	


def test_model(model, loss_function, data_loader):
	# set model in evaluation mode
	model.eval()
	current_loss = 0.0
	current_acc = 0
	# iterate over the validation data
	for i, (inputs, labels) in enumerate(data_loader):
		# send the input/labels to the GPU
		#inputs = torch.Tensor(inputs).to(device)
		#labels = torch.Tensor(labels).to(device)
		# forward
		with torch.set_grad_enabled(False):
			outputs = model(inputs,inputs.shape[1])
			_, predictions = torch.max(outputs, 1)
			loss = loss_function(outputs, labels)
			# statistics
		current_loss += loss.item() * inputs.size(0)
		current_acc += torch.sum(predictions == labels.data)
	total_loss = current_loss / len(data_loader.dataset)
	total_acc = current_acc.double() / len(data_loader.dataset)
	print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss,
	total_acc))
	return total_loss, total_acc


class LSTMModel(torch.nn.Module):
	def __init__(self, vocab_size, embedding_size, hidden_size,output_size, pad_idx):
		super().__init__()
		# Embedding field
		self.embedding=torch.nn.Embedding(num_embeddings=vocab_size,
		embedding_dim=embedding_size,padding_idx=pad_idx)
		# LSTM cell
		self.rnn = torch.nn.LSTM(input_size=embedding_size,hidden_size=hidden_size)
		# Fully connected output
		self.fc = torch.nn.Linear(hidden_size, output_size)
	def forward(self, text_sequence, text_lengths):
		# Extract embedding vectors
		embeddings = self.embedding(text_sequence)
		# Pad the sequences to equal length
		packed_sequence =torch.nn.utils.rnn.pack_padded_sequence(embeddings, text_lengths)
		packed_output, (hidden, cell) = self.rnn(packed_sequence)
		return self.fc(hidden)


if __name__ == '__main__':
	tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
	TEXT = torchtext.data.Field(
	tokenize=tokenizer,
	lower=True, # convert all letters to lower case
	include_lengths=True, # include the length of the movie review
	)
	LABEL = torchtext.data.LabelField(dtype=torch.float)
	train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL)
	TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(name='6B',dim=100))
	LABEL.build_vocab(train)
	train_iter, test_iter = torchtext.data.BucketIterator.splits((train, test), sort_within_batch=True, batch_size=64,device=device)
	model = LSTMModel(vocab_size=len(TEXT.vocab),embedding_size=EMBEDDING_SIZE,hidden_size=HIDDEN_SIZE,output_size=1,pad_idx=TEXT.vocab.stoi[TEXT.pad_token])
	model.embedding.weight.data.copy_(TEXT.vocab.vectors)
	model.embedding.weight.data[TEXT.vocab.stoi[TEXT.unk_token]] = torch.zeros(EMBEDDING_SIZE)
	model.embedding.weight.data[TEXT.vocab.stoi[TEXT.pad_token]] = torch.zeros(EMBEDDING_SIZE)
	optimizer = torch.optim.Adam(model.parameters())
	loss_function = torch.nn.BCEWithLogitsLoss()
	for epoch in range(100):
		print(f"Epoch {epoch + 1}/5")
		train_model(model, loss_function, optimizer, train_iter)
		test_model(model, loss_function, test_iter)
