import csv
import numpy as np
import nltk
import copy
import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('punkt')
nltk.download('stopwords')

class ELMoTrainer:
    # defining the ELMo model
    class ELMo(nn.Module):
        def __init__(self, vocab_size, embed_size, embeddings_vector, device):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed_size = embed_size
            self.embeddings_vector = embeddings_vector
            self.device = device

            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.lstm1_forward = nn.LSTM(self.embed_size, self.embed_size // 2, batch_first=True, bidirectional=False)
            self.lstm1_backward = nn.LSTM(self.embed_size, self.embed_size // 2, batch_first=True, bidirectional=False)
            self.lstm2_forward = nn.LSTM(self.embed_size // 2, self.embed_size // 2, batch_first=True, bidirectional=False)
            self.lstm2_backward = nn.LSTM(self.embed_size // 2, self.embed_size // 2, batch_first=True, bidirectional=False)

            self.linear = nn.Linear(self.embed_size, self.embed_size)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(self.embed_size, self.vocab_size)

            # load the embeddings into the model
            self.embed.weight.data.copy_(torch.tensor(self.embeddings_vector))

        def forward(self, x, mode='train', ls='const'):
            if mode == 'train':
                x1 = self.embed(x)
                x2_forward, _ = self.lstm1_forward(x1)
                x2_backward, _ = self.lstm1_backward(x1.flip(1))
                x3_forward, _ = self.lstm2_forward(x2_forward)
                x3_backward, _ = self.lstm2_backward(x2_backward)
                x3_for_padded = torch.cat((torch.zeros(x3_forward.shape[0], 1, x3_forward.shape[2]).to(self.device), x3_forward[:, :-1, :]), dim=1)
                x3_back_padded = torch.cat((x3_backward.flip(1)[:, 1:, :], torch.zeros(x3_backward.shape[0], 1, x3_backward.shape[2]).to(self.device)), dim=1)
                x4 = torch.cat((x3_for_padded, x3_back_padded), dim=2)
                x5 = self.linear(x4)
                x6 = self.relu(x5)
                x7 = self.linear2(x6)
                return x7

            else:
                x1 = self.embed(x)
                x2_forward, _ = self.lstm1_forward(x1)
                x2_backward, _ = self.lstm1_backward(x1.flip(1))
                x3_forward, _ = self.lstm2_forward(x2_forward)
                x3_backward, _ = self.lstm2_backward(x2_backward)
                e0 = x1
                e1 = torch.cat((x2_forward, x2_backward.flip(1)), dim=2)
                e2 = torch.cat((x3_forward, x3_backward.flip(1)), dim=2)

                return e0, e1, e2

    # defining the RNN classifier with torch
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, type):
            super().__init__()

            self.type = type

            if type == 'const':
                pass
            elif type == 'learnable':
                self.learnable_scalar = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]))
            elif type == 'learnable_func':
                self.learnable_func1 = nn.Linear(3*input_size, 4*input_size)
                self.learnable_relu = nn.ReLU()
                self.learnable_func2 = nn.Linear(4*input_size, input_size)

            self.hidden_size = hidden_size
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc1 = nn.Linear(hidden_size, hidden_size//2)
            self.fc2 = nn.Linear(hidden_size//2, output_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            e0, e1, e2 = x
            if self.type == 'const':
                e = (e0 + e1 + e2) / 3
            elif self.type == 'learnable':
                e = self.learnable_scalar[0] * e0 + self.learnable_scalar[1] * e1 + self.learnable_scalar[2] * e2
            elif self.type == 'learnable_func':
                e = torch.cat((e0, e1, e2), dim=2)
                e = self.learnable_func1(e)
                e = self.learnable_relu(e)
                e = self.learnable_func2(e)

            out, _ = self.rnn(e)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    # defining the NewsDataset class for the language modeling task
    class NewsDatasetLanguageModeling(Dataset):
        def __init__(self, ids_x):
            self.ids = ids_x

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            return self.ids[idx]

    # defining the NewsDataset class for the classification task
    class NewsDatasetClassification(Dataset):
        def __init__(self, ids_x, labels):
            self.ids_x = ids_x
            self.labels = labels

        def __len__(self):
            return len(self.ids_x)

        def __getitem__(self, idx):
            return self.ids_x[idx], self.labels[idx]

    def __init__(
        self,
        train_data=None,
        test_data=None,
        batch_size=128,
        embed_size=200,
        learning_rate=1e-3,
        epochs=10,
        save_model=False,
        load_model=False,
        model_path='./model.pth',
        classify=False,
        ls='const',
        hidden_size=100,
        classifier_epochs=5,
        batch_size_classifier=128,
        save_model_classifier=False,
        model_path_classifier='./model_classifier.pth'):

        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_model = save_model
        self.load_model = load_model
        self.model_path = model_path
        self.classify = classify
        self.ls = ls
        self.hidden_size = hidden_size
        self.classifier_epochs = classifier_epochs
        self.batch_size_classifier = batch_size_classifier
        self.save_model_classifier = save_model_classifier
        self.model_path_classifier = model_path_classifier

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('Preparing data...')
        self.prepare_data()
        print('Data prepared')

        if self.load_model == False:
            print('Creating model...')
            self.create_model()
            print('Model created')

            print('Training model...')
            for epoch in range(self.epochs):
                self.train_lm(epoch)
                self.test_lm()
            print('Model trained')

            if self.save_model:
                torch.save(self.model.state_dict(), self.model_path)
                print('Model saved')
        else:
            print('Loading model...')
            self.create_model()
            self.model.load_state_dict(torch.load(self.model_path))
            print('Model loaded')

            print('Testing model...')
            self.test_lm()
            print('Model tested')

        if self.classify:
            print('Preparing data for classification...')
            self.prepare_data_classification()
            print('Data prepared')

            print('Creating classifier...')
            self.create_classifier()
            print('Classifier created')

            print('Training classifier...')
            for epoch in range(self.classifier_epochs):
                self.train_classifier(epoch)
                self.test_classifier()
            print('Classifier trained')

            if self.save_model_classifier:
                torch.save(self.classifier.state_dict(), self.model_path_classifier)
                print('Classifier saved')

    def prepare_data(self):
        print('    Tokenising data...')
        # tokenising the train data to get the individual words
        train_data_description = [row[1] for row in self.train_data]
        train_data_tokenised = [nltk.word_tokenize(description) for description in train_data_description]
        test_data_description = [row[1] for row in self.test_data]
        test_data_tokenised = [nltk.word_tokenize(description) for description in test_data_description]
        # tokenise the words based on '\\' as well
        train_data_tokenised = [[word.split('\\') for word in description] for description in train_data_tokenised]
        test_data_tokenised = [[word.split('\\') for word in description] for description in test_data_tokenised]

        # print all the elements in the list within the tokenised list
        train_data_temp = copy.deepcopy(train_data_tokenised)
        test_data_temp = copy.deepcopy(test_data_tokenised)
        train_data_tokenised = []
        test_data_tokenised = []
        for description in train_data_temp:
            train_data_tokenised.append([])
            for words in description:
                for word in words:
                    if word != '':
                        train_data_tokenised[-1].append(word)
        for description in test_data_temp:
            test_data_tokenised.append([])
            for words in description:
                for word in words:
                    if word != '':
                        test_data_tokenised[-1].append(word)

        # convert all the words to lower case
        self.train_data_tokenised = [[word.lower() for word in description] for description in train_data_tokenised]
        self.test_data_tokenised = [[word.lower() for word in description] for description in test_data_tokenised]

        print('    Data tokenised')

        # finding the unique words in the train data
        embeddings = torch.load('Data/skip-gram-word-vectors.pt')
        vocab = list(embeddings.keys())
        vocab.append('<pad>')
        embeddings['<pad>'] = np.zeros(200)
        self.vocab = sorted(vocab)
        self.vocab_size = len(vocab)
        self.embeddings_vector = np.array([embeddings[word] for word in vocab])

        # making a dictionary of the words and their corresponding indices
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        train_data_lens = [len(description) for description in self.train_data_tokenised]
        test_data_lens = [len(description) for description in self.test_data_tokenised]

        train_data_len = np.percentile(train_data_lens, 90)
        test_data_len = np.percentile(test_data_lens, 90)

        # trim all the data to max length
        self.train_data_tokenised = [description[:int(train_data_len)] for description in self.train_data_tokenised]
        self.test_data_tokenised = [description[:int(test_data_len)] for description in self.test_data_tokenised]

        print('    Padding data...')

        train_data_ids = [[self.word2idx.get(word, self.word2idx['<oov>']) for word in description] for description in self.train_data_tokenised]
        test_data_ids = [[self.word2idx.get(word, self.word2idx['<oov>']) for word in description] for description in self.test_data_tokenised]

        train_data_pad = []
        test_data_pad = []
        for description in train_data_ids:
            if len(description) >= train_data_len:
                train_data_pad.append(description[:int(train_data_len)])
            else:
                train_data_pad.append(description + [self.word2idx['<pad>']] * int(train_data_len - len(description)))
        for description in test_data_ids:
            if len(description) >= test_data_len:
                test_data_pad.append(description[:int(test_data_len)])
            else:
                test_data_pad.append(description + [self.word2idx['<pad>']] * int(test_data_len - len(description)))

        print('    Data padded')

        train_data_pad = np.array(train_data_pad)
        test_data_pad = np.array(test_data_pad)

        train_data_pad = torch.tensor(train_data_pad)
        test_data_pad = torch.tensor(test_data_pad)

        train_data_pad_x = train_data_pad[:, :-1]
        test_data_pad_x = test_data_pad[:, :-1]

        # converting the data to tensors
        self.train_data_pad_x = torch.tensor(train_data_pad_x)
        self.test_data_pad_x = torch.tensor(test_data_pad_x)

        # creating the dataset
        self.train_dataset = self.NewsDatasetLanguageModeling(self.train_data_pad_x)
        self.test_dataset = self.NewsDatasetLanguageModeling(self.test_data_pad_x)

        # creating the dataloaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def create_model(self):
        self.model = self.ELMo(self.vocab_size, self.embed_size, self.embeddings_vector, self.device).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_lm(self, epoch):
        self.model.train()

        running_loss = 0.0
        total = 0
        accs = []
        pre = []
        rec = []
        f1 = []

        print(f'Epoch {epoch + 1}/{self.epochs}')
        for idx, data in enumerate(tqdm.tqdm(self.train_dataloader)):
            data_x = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data_x)
            loss = self.criterion(output.view(-1, self.vocab_size), data_x.view(-1))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            total += data_x.size(0)
            accs.append(accuracy_score(data_x.cpu().numpy().flatten(), output.argmax(dim=-1).cpu().numpy().flatten()))
            pre.append(precision_score(data_x.cpu().numpy().flatten(), output.argmax(dim=-1).cpu().numpy().flatten(), average='weighted', zero_division=0))
            rec.append(recall_score(data_x.cpu().numpy().flatten(), output.argmax(dim=-1).cpu().numpy().flatten(), average='weighted', zero_division=0))
            f1.append(f1_score(data_x.cpu().numpy().flatten(), output.argmax(dim=-1).cpu().numpy().flatten(), average='weighted', zero_division=0))

        print(f'Loss: {running_loss / len(self.train_dataloader)}')
        print(f'Accuracy: {np.mean(accs)}')
        print(f'Precision: {np.mean(pre)}')
        print(f'Recall: {np.mean(rec)}')
        print(f'F1: {np.mean(f1)}')

    def test_lm(self):
        self.model.eval()

        running_loss = 0.0
        total = 0
        accs = []
        pre = []
        rec = []
        f1 = []

        with torch.no_grad():
            for idx, data in enumerate(tqdm.tqdm(self.test_dataloader)):
                data_x = data.to(self.device)
                output = self.model(data_x)
                loss = self.criterion(output.view(-1, self.vocab_size), data_x.view(-1))
                running_loss += loss.item()
                total += data_x.size(0)
                accs.append(accuracy_score(data_x.cpu().numpy().flatten(), output.argmax(dim=-1).cpu().numpy().flatten()))
                pre.append(precision_score(data_x.cpu().numpy().flatten(), output.argmax(dim=-1).cpu().numpy().flatten(), average='weighted', zero_division=0))
                rec.append(recall_score(data_x.cpu().numpy().flatten(), output.argmax(dim=-1).cpu().numpy().flatten(), average='weighted', zero_division=0))
                f1.append(f1_score(data_x.cpu().numpy().flatten(), output.argmax(dim=-1).cpu().numpy().flatten(), average='weighted', zero_division=0))

        print(f'Loss: {running_loss / len(self.test_dataloader)}')
        print(f'Accuracy: {np.mean(accs)}')
        print(f'Precision: {np.mean(pre)}')
        print(f'Recall: {np.mean(rec)}')
        print(f'F1: {np.mean(f1)}')

    def prepare_data_classification(self):
        train_data_labels = [row[0] for row in self.train_data]
        test_data_labels = [row[0] for row in self.test_data]

        train_data_labels = [int(label) for label in train_data_labels]
        test_data_labels = [int(label) for label in test_data_labels]

        # find the unique labels
        labels = list(set(train_data_labels))
        labels = sorted(labels)

        self.labels_count = len(labels)

        label2idx = {label: idx for idx, label in enumerate(labels)}

        # one hot encoding the labels
        self.train_data_labels = torch.zeros(len(train_data_labels), self.labels_count)
        self.test_data_labels = torch.zeros(len(test_data_labels), self.labels_count)
        for idx, label in enumerate(train_data_labels):
            self.train_data_labels[idx, label2idx[label]] = 1
        for idx, label in enumerate(test_data_labels):
            self.test_data_labels[idx, label2idx[label]] = 1

        self.train_dataset_classification = self.NewsDatasetClassification(self.train_data_pad_x, self.train_data_labels)
        self.test_dataset_classification = self.NewsDatasetClassification(self.test_data_pad_x, self.test_data_labels)

        self.train_dataloader_classification = DataLoader(self.train_dataset_classification, batch_size=self.batch_size_classifier, shuffle=True)
        self.test_dataloader_classification = DataLoader(self.test_dataset_classification, batch_size=self.batch_size_classifier, shuffle=False)

    def create_classifier(self):
        self.classifier = self.RNN(self.embed_size, self.hidden_size, self.labels_count, self.ls).to(self.device)
        self.criterion_classifier = nn.CrossEntropyLoss()
        self.optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

    def train_classifier(self, epoch):
        self.model.eval()
        self.classifier.train()

        running_loss = 0.0
        total = 0
        accs = []
        pre = []
        rec = []
        f1 = []

        data_ys = []
        actuals = []

        print(f'Epoch {epoch + 1}/{self.classifier_epochs}')
        for idx, data in enumerate(tqdm.tqdm(self.train_dataloader_classification)):
            data_x, data_y = data
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            self.optimizer_classifier.zero_grad()
            with torch.no_grad():
                output = self.model(data_x, mode='test', ls=self.ls)
            output = self.classifier(output)
            output = torch.sum(output, dim=1)
            loss = self.criterion_classifier(output, data_y)
            loss.backward()
            self.optimizer_classifier.step()
            running_loss += loss.item()
            total += data_x.size(0)
            accs.append(accuracy_score(data_y.argmax(dim=-1).cpu().numpy(), output.argmax(dim=-1).cpu().numpy()))
            pre.append(precision_score(data_y.argmax(dim=-1).cpu().numpy(), output.argmax(dim=-1).cpu().numpy(), average='weighted', zero_division=0))
            rec.append(recall_score(data_y.argmax(dim=-1).cpu().numpy(), output.argmax(dim=-1).cpu().numpy(), average='weighted', zero_division=0))
            f1.append(f1_score(data_y.argmax(dim=-1).cpu().numpy(), output.argmax(dim=-1).cpu().numpy(), average='weighted', zero_division=0))
            data_ys.extend(data_y.argmax(dim=-1).cpu().numpy())
            actuals.extend(output.argmax(dim=-1).cpu().numpy())
        # # confusion matrix
        # plt.figure(figsize=(10, 10))
        # plt.imshow(data_ys, actuals)
        # plt.savefig(f'confusion_matrix_{epoch}.png')

        print(f'Loss: {running_loss / len(self.train_dataloader_classification)}')
        print(f'Accuracy: {np.mean(accs)}')
        print(f'Precision: {np.mean(pre)}')
        print(f'Recall: {np.mean(rec)}')
        print(f'F1: {np.mean(f1)}')

    def test_classifier(self):
        self.model.eval()
        self.classifier.eval()

        running_loss = 0.0
        total = 0
        accs = []
        pre = []
        rec = []
        f1 = []

        with torch.no_grad():
            for idx, data in enumerate(tqdm.tqdm(self.test_dataloader_classification)):
                data_x, data_y = data
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                output = self.model(data_x, mode='test', ls=self.ls)
                output = self.classifier(output)
                output = torch.sum(output, dim=1)
                loss = self.criterion_classifier(output, data_y)
                running_loss += loss.item()
                total += data_x.size(0)
                accs.append(accuracy_score(data_y.argmax(dim=-1).cpu().numpy(), output.argmax(dim=-1).cpu().numpy()))
                pre.append(precision_score(data_y.argmax(dim=-1).cpu().numpy(), output.argmax(dim=-1).cpu().numpy(), average='weighted', zero_division=0))
                rec.append(recall_score(data_y.argmax(dim=-1).cpu().numpy(), output.argmax(dim=-1).cpu().numpy(), average='weighted', zero_division=0))
                f1.append(f1_score(data_y.argmax(dim=-1).cpu().numpy(), output.argmax(dim=-1).cpu().numpy(), average='weighted', zero_division=0))

        print(f'Loss: {running_loss / len(self.test_dataloader_classification)}')
        print(f'Accuracy: {np.mean(accs)}')
        print(f'Precision: {np.mean(pre)}')
        print(f'Recall: {np.mean(rec)}')
        print(f'F1: {np.mean(f1)}')


if __name__ == '__main__':
    # loading the data
    train_data = None
    test_data = None
    with open('Data/train.csv', 'r') as file:
        train_data = list(csv.reader(file))
        train_data = train_data[1:] # remove the header
    with open('Data/test.csv', 'r') as file:
        test_data = list(csv.reader(file))
        test_data = test_data[1:] # remove the header

    # training the model
    params = {
        'train_data': train_data,
        'test_data': test_data,
        'batch_size': 128,
        'embed_size': 200,
        'learning_rate': 1e-3,
        'epochs': 5,
        'save_model': False,
        'load_model': True,
        'model_path': 'model2.pth',
        'classify': False,
        'ls': 'learnable_func',
        'hidden_size': 100,
        'classifier_epochs': 15,
        'batch_size_classifier': 128,
        'save_model_classifier': True,
        'model_path_classifier': 'model_classifier3.pth'
    }

    trainer = ELMoTrainer(**params)
