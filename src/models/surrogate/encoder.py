from os.path import exists

import torch
from torch import nn
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.nn import global_max_pool
import torch.nn.functional as F
from gensim.models import KeyedVectors

from src.vocabulary import Vocabulary
from src.models.surrogate.gcn import GNN


class SurrogateEncoder(torch.nn.Module):
    """
    Simple GCN encoder for later explanation with LRP.
    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int, device: str = 'cpu'):
        super(SurrogateEncoder, self).__init__()
        self.device = device
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = CustomSTEncoder(config, vocab, vocabulary_size, pad_idx, device)

        self.gcn_layers = []
        if config.n_hidden_layers > 0:
            for i in range(config.n_hidden_layers):
                input_size = config.rnn.hidden_size if i == 0 else config.hidden_size
                setattr(self, f"gcn_layer{i}",
                        GNN(input_size, config.hidden_size))
        self.gcn_layers = torch.nn.ModuleList(self.gcn_layers)

    def forward(self, batched_graph: Batch):
        batch = batched_graph.batch
        edge_index = batched_graph.edge_index

        # initial node embedding
        out = self.__st_embedding(batched_graph.x)

        # GCN layers
        if self.__config.n_hidden_layers > 0:
            for i in range(self.__config.n_hidden_layers - 1):
                out = F.relu(getattr(self, f"gcn_layer{i}")(out, edge_index))

        # graph-level pooling
        out = global_max_pool(out, batch)
        return out


class CustomSTEncoder(torch.nn.Module):
    """

    encoder for statement without recurrence

    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int, device: str = 'cpu'):
        super(CustomSTEncoder, self).__init__()
        self.device = device
        self.__pad_idx = pad_idx
        if exists(config.w2v_path):
            self.__add_w2v_weights(config.w2v_path, vocab)

    def __add_w2v_weights(self, w2v_path: str, vocab: Vocabulary):
        """
        add pretrained word embedding to embedding layer

        Args:
            w2v_path: path to the word2vec model

        Returns:

        """
        model = KeyedVectors.load(w2v_path, mmap="r")
        #model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)

        weights = torch.FloatTensor(model.vectors, device=self.device)
        # TODO: compensate for <UNK> token -> zero embedding
        unk_weights = torch.zeros((1, weights.size(1)), device=self.device)
        weights = torch.cat((unk_weights, weights))
        self.__wd_embedding = nn.Embedding.from_pretrained(weights, padding_idx=self.__pad_idx, freeze=False)

    def forward(self, seq: torch.Tensor):
        """

        Args:
            seq: [n nodes (seqs); max parts (seq len); embed dim]

        Returns:

        """
        # [n nodes; max parts; embed dim]
        wd_embedding = self.__wd_embedding(seq)
        # [n nodes; embed dim]
        wd_embedding = torch.sum(wd_embedding, dim=1)
        # [n nodes; rnn hidden]
        node_embedding = wd_embedding
        return node_embedding
