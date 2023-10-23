import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
# from LRU_pytorch import LRU
# from indrnn import IndRNNv2
import torch.optim as optim
from glob import glob
from torch.utils.tensorboard import SummaryWriter


from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

GPU_TYPE = {
    "A40": 0,
    "2080Ti": 1,
    "RTX_TITAN": 2,
    "TITAN_Xp": 3,
    "TITAN_V": 4,
    "1080": 5,
    "TITAN_X": 6,
    "M40": 7
}

class ProfileDataset(Dataset):
    def __init__(self, data_path, meta_path, mask_path, feature_path):
        try:
            import pandas as pd
            import numpy as np
            import random
        except:
            raise("pandas or numpy not found")
        try:
            self.data = pd.read_pickle(data_path)
            self.meta = pd.read_pickle(meta_path)
            self.mask = pd.read_pickle(mask_path)
            self.feature = pd.read_pickle(feature_path)
        except:
            raise("load data failed")
        self.seed = 19260817
        # random.seed(self.seed)
        # only_one_model_batch = []
        # for k,v in self.feature.items():
        #     if sum(v - np.ones(8)<0) < 2:
        #         only_one_model_batch.append(k)
        # select_from_meta = []
        # for model, batch in only_one_model_batch:
        #     s = self.meta[(self.meta.model == model) & (self.meta.batch == batch)]
        #     select_from_meta.append(s)
        # for x in select_from_meta:
        #     self.meta.drop(x.index, inplace=True)
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        gpu, model, batch, length, time_a, gpu_mem_a, cpu_mem_a = self.meta.iloc[idx]
        name_path = "{}_{}_{}".format(gpu, model, batch)
        raw_data = self.data[name_path]
        mask_vector = self.mask[name_path]
        kernel = raw_data[:, 0]
        vector = raw_data[:, 1:13]
        vector = vector.astype(np.float32)
        feature_vec = self.feature[(model, batch)]
        feature_vec = feature_vec.astype(np.float32) / np.max(feature_vec)
        return kernel, vector, feature_vec, mask_vector, length
    
class RandomMaskModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, mlp_input_size, mlp_hidden_sizes):
        super(RandomMaskModule, self).__init__()
        self.device_indicator = nn.Parameter(torch.empty(0))
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        
        layers = [nn.Linear(mlp_input_size, mlp_hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(mlp_hidden_sizes)):
            layers.extend([nn.Linear(mlp_hidden_sizes[i - 1], mlp_hidden_sizes[i]), nn.ReLU()])
        layers.extend([nn.Linear(mlp_hidden_sizes[-1], embedding_dim), nn.ReLU()])
        self.mlp = nn.Sequential(*layers)

    def forward(self, number_seq_input, vec_seq_input, mask, vector_rate=0.2):
        extra_mask = torch.rand(number_seq_input.shape).to(self.device_indicator.device)
        extra_mask = extra_mask < vector_rate
        # amsk is extra_mask or mask
        mask = extra_mask | mask

        embedded_numbers = self.embedding_layer(number_seq_input)
        masked_embedded_numbers = embedded_numbers * (1 - mask.unsqueeze(-1).float())

        mlp_output = self.mlp(vec_seq_input)
        masked_mlp_output = mlp_output * mask.unsqueeze(-1).float()

        output = masked_embedded_numbers + masked_mlp_output

        return output
    
class transformer2vector(nn.Module):
    # def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout):
    def __init__(self, input_size=32, hidden_size=128, layer_num=2, kernel_num=1182, feature_size=8, num_heads=16, dropout=0.2):

        super(transformer2vector, self).__init__()
        self.mask_module = RandomMaskModule(kernel_num, input_size, 12, [32, 64])
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_encoder_layer = TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, layer_num)
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, feature_size)
        
    def forward(self, number_seq_input, vec_seq_input, mask, length):
        embedded_seq = self.mask_module(number_seq_input, vec_seq_input, mask)
        # packed_seq = torch.nn.utils.rnn.pack_padded_sequence(embedded_seq, length, batch_first=True, enforce_sorted=False)
        transformer_output = self.transformer_encoder(embedded_seq)
        # Take the mean of transformer output across all positions in the sequence
        seq_vector = torch.mean(transformer_output, dim=1)
        output = self.fc1(seq_vector)
        output = self.fc2(output)
        output = self.fc3(output)
        return output
    
if __name__ == '__main__':
    writer = SummaryWriter()
    profile_dataset = ProfileDataset("data.pkl", "meta.pkl", "mask.pkl", "feature.pkl")
    train_size = int(0.8 * len(profile_dataset))
    test_size = len(profile_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(profile_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=70, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=70, shuffle=True, num_workers=4)
    model = transformer2vector()
    
    # last_model_num = sorted([int(x.split('_')[-1].split('.')[0]) for x in glob("model_*.pt")])[-1]

    # print(last_model_num)

    model.load_state_dict(torch.load("correct_model_50.pt"))

    model = model.cuda()
    criterion = nn.SmoothL1Loss(reduction='sum')
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    for epoch in range(10000):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            kernel, vector, feature_vec, mask, length = data
            kernel = kernel.cuda()
            vector = vector.cuda()
            feature_vec = feature_vec.cuda()
            mask = mask.cuda()
            length = length.cuda()
            optimizer.zero_grad()
            output = model(kernel, vector, mask, length)
            output = output * (feature_vec > 0)
            output_max, _ = output.max(dim=1, keepdim=True)
            output = output / output_max
            loss = criterion(output, feature_vec)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / (i + 1)))
        writer.add_scalar('training loss',
                            running_loss / (i + 1),
                            epoch * len(train_dataloader) + i)
        running_loss = 0.0
        #test
        test_loss = 0.0
        for i, data in enumerate(test_dataloader, 0):
            kernel, vector, feature_vec, mask, length = data
            kernel = kernel.cuda()
            vector = vector.cuda()
            feature_vec = feature_vec.cuda()
            mask = mask.cuda()
            length = length.cuda()
            output = model(kernel, vector, mask, length)
            # where feature vec is zero then set output to zero
            output = output * (feature_vec > 0)
            output_max, _ = output.max(dim=1, keepdim=True)
            output = output / output_max
            loss = criterion(output, feature_vec)
            test_loss += loss.item()
        print("test loss: %.3f" % (test_loss / len(test_dataloader)))
        writer.add_scalar('test loss',
                            test_loss / len(test_dataloader),
                            epoch)
        #save
        if epoch % 10 == 9:
            torch.save(model.state_dict(), "correct_model_{}.pt".format(epoch + 1 + 50))


    writer.close()
