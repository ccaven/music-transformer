
import pickle
import math
import torch
import numpy as np
import os

import pypianoroll as piano

import matplotlib.pyplot as plt

from data import SongDataset, get_input_and_target
from model import Transformer

CUDA = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

EPOCHS = 2500

def load_new(ntoken):
    return Transformer(ntoken=ntoken,
                        ninp=512,
                        nhead=32,
                        nhid=40,
                        nlayers=16,
                        dropout=0.2)

def load_presaved(f):
    with open(f, "rb") as handle:
        model = torch.load(handle)
    return model

def train(model: Transformer, song_batches, ntoken):
    model = model.to(CUDA)
    
    #opt = torch.optim.Adam(model.parameters(), lr=0.01)
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
    
    # Train
    
    total_loss_list = []
    for epoch in range(1, EPOCHS):  
        total_loss = 0       
        
        # Loop through batches
        for index, batch in enumerate(song_batches):
            # Get input/target pairs for batch
            inputs, targets = get_input_and_target(batch)
            
            # Send tensor to GPU
            inputs = inputs.to(CUDA)
            
            # Prepare optimizer
            opt.zero_grad()
            
            # Make prediction
            pred: torch.Tensor = model(inputs).to(CUDA)     
            
            # unwrap to format for nn.CrossEntropyLoss
            targets = targets.type(torch.int64).to(CUDA).view(-1) 
            pred = pred.view(-1, ntoken) 
            
            # Run loss
            loss = loss_fn(pred, targets)
            loss.backward()
            
            # Step optimizer
            opt.step()
            
            # Stats
            print(epoch, index, "/", len(song_batches), ":", loss.detach().item())
            total_loss += loss.detach().item()
            
        total_loss /= len(song_batches)
        total_loss_list.append(total_loss)
        
        # Update learning rate
        scheduler.step()
        
        if epoch % 100 == 0:
            torch.save(model, "models/5/" + str(2500+epoch) + "_epochs.pth")
        
        print("\n")
        print(epoch, ":", total_loss)
        print("~" * 20)
        
    # Save model
    torch.save(model, "models/5/" + str(2500+EPOCHS) + "_epochs.pth")
    
    plt.plot(total_loss_list)
    plt.xlabel("Iter")
    plt.ylabel("Loss")
    plt.show()

def predict(model: Transformer, dataset: SongDataset, song_batches, ntoken):
    i = 0
    model.to(CPU)
    for idx, song in enumerate(dataset.songs):
        
        if len(song) < 128: continue
        
        seq = song[:128]
              
        input_seq = np.array([seq])
        input_tensor = torch.tensor(input_seq, device=torch.device("cpu"), dtype=torch.int32)
        
        #print(input_tensor.shape)
        pred: torch.Tensor = model(input_tensor)
        pred = pred.cpu().topk(1)[1].view(-1).tolist()
        
        # Convert numbers to lists of notes        
        notes = [dataset.dictionary.wordFromIndex(i) for i in pred]
        
        notes = [dataset.dictionary.listFromWord(i) if "|" in i else [] for i in notes]
        
        # Turn into piano roll
        piano_roll = [[0 for j in range(128)] for i in range(len(notes))]
        
        for i, notelist in enumerate(notes):
            for note in notelist:
                piano_roll[i][note] = 1
        
        piano_roll = np.array(piano_roll)
        
        # Turn into pypianoroll track
        track = piano.Track("track", 0, False, piano_roll).binarize()
        multitrack = piano.Multitrack("tracks", tempo=np.ones(len(notes)) * 15, tracks=[track])
        
        piano.plot(multitrack)
        plt.savefig("generated/7-png/" + str(idx) + ".png")
        #plt.show()
        
        midi = multitrack.to_pretty_midi()
        midi.write("generated/7-mid/" + str(idx) + ".mid")
        
        print("Wrote", idx, "mid and png to generated/6/")

def get_total_loss_grapb(dataset: SongDataset, song_batches, ntoken):
    # find all model files
    all_files = os.listdir("models/5")
    loss_fn = torch.nn.CrossEntropyLoss()
    
    total_loss_x = []
    total_loss_y = []
    
    for fname in all_files:
        # Load model
        model: Transformer = load_presaved("models/5/" + fname)
        #model = model.cpu()

        print("Loading", fname)
        
        # Find total loss
        total_loss = 0
        for batch in song_batches:
            inputs, targets = get_input_and_target(batch)
            
            # Send tensor to GPU
            inputs = inputs.cuda()
            
            # Make prediction
            pred: torch.Tensor = model(inputs)
            
            # unwrap to format for nn.CrossEntropyLoss
            targets = targets.type(torch.int64).cuda().view(-1) 
            pred = pred.view(-1, ntoken)
            
            # Run loss
            loss = loss_fn(pred, targets)
            total_loss += loss.detach().item()
            
            del inputs
            del targets
            del pred
            torch.cuda.synchronize()
        
        total_loss /= len(song_batches)
        print("Total loss", total_loss)    
        
        total_loss_x.append(int(fname.split("_")[0]))
        total_loss_y.append(total_loss)
        
        del model
        torch.cuda.synchronize()
    
    with open("stats/total_loss.pickle", "wb") as handle:
        pickle.dump([total_loss_x, total_loss_y], handle)
    
    
    
    print(all_files)
    
    return

def main():
    
    #data = SongDataset()
    #data.load_pickle("data/Nottingham.pickle")
    #song_batches = data.create_batches(16, 128)
    
    #ntoken = len(data.dictionary)
    
    #print("Num tokens: ", ntoken)
    
    #model = load_new(ntoken=ntoken)
    #model = load_presaved("models/5/5000_epochs.pth")
    
    #predict(model, data, song_batches, ntoken)
    #train(model, song_batches, ntoken)

    #get_total_loss_grapb(data, song_batches, ntoken)

    with open("stats/total_loss.pickle", "rb") as f:
        x, y = pickle.load(f)
        d = zip(x, y)
        d = sorted(d, key=lambda a: a[0])
        x = [i[0] for i in d]
        y = [i[1] for i in d]
        plt.plot(x, y)
        #plt.xscale("log")
        plt.title("Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    return
    

if __name__ == "__main__":
    main()