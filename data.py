
import pickle
import os
import numpy as np
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        
    def __len__(self):
        return len(self.idx2word)
    
    def wordFromIndex(self, index: int):
        assert index >= 0 and index < len(self)
        return self.idx2word[index]
    
    def indexFromWord(self, word):
        assert word in self.word2idx
        return self.word2idx[word]
    
    def wordFromList(self, list: list):
        return "|".join(map(str, list))
    
    def listFromWord(self, word: str):
        return [int(i) for i in word.split("|")]
    
    def add_word(self, word):
        if word not in self.word2idx:
            print("added word", word)
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

class SongDataset(object):
    def __init__(self,):
        self.dictionary = Dictionary()
        self.songs = []
    
    def load_pickle(self, filename: str):
        assert os.path.exists(filename)
        
        with open(filename, "rb") as f:
            raw_data = pickle.load(f)                                   
            
        raw_data = raw_data["train"] + raw_data["test"] + raw_data["valid"]
        
        self.dictionary.add_word("<sos>")
        self.dictionary.add_word("<eos>")
        
        for song in raw_data:
            song = ["<sos>"] + [self.dictionary.wordFromList(i) for i in song] + ["<eos>"]            
            new_song = []
            for note in song:
                new_song.append(self.dictionary.add_word(note))
            self.songs.append(new_song)
    
    def create_batches(self, batch_size: int, seq_len: int):
        # Split into mini sequences
        seqs = []        
        for song in self.songs:
            if len(song) < seq_len: continue
            for index in range(0, len(song) - seq_len, seq_len // 2):
                seq = song[index : index + seq_len]
                seqs.append(seq)
        
        seqs = np.array(seqs)
        np.random.shuffle(seqs)
        
        # Split into batches
        batches = []
        for index in range(0, len(seqs), batch_size):
            if index + batch_size <= len(seqs):
                batches.append(seqs[index : index + batch_size])
        
        batches = np.array(batches)
        
        return torch.tensor(batches, dtype=torch.int32)

def get_input_and_target (batch: torch.Tensor):
    # Split seq in half
    # Transformer tries to predict the other half
    
    # Other option is for the transformer to predict the last note
    
    l = batch.size(1) // 2
    inputs = batch[:, :l]
    outputs = batch[:, l:l*2]
    return inputs, outputs

def main():
    
    dataset = SongDataset()
    dataset.load_pickle("data/Nottingham.pickle")
    
    song_batches = dataset.create_batches(32, 128)
    
    print(song_batches.size())
    
    print(len(dataset.dictionary))
    
    inputs, outputs = get_input_and_target(song_batches[0])
    
    print(inputs.size(), outputs.size())
    
    return

if __name__ == "__main__":
    main()