import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange
import datetime


def save_model(epochs, model, loss,act1,act2, batch_size, integrator, sys,shape):
    path = f"Models/{sys}_{integrator}_{epochs}epoch_{act1}_{act2}_batchsize_{batch_size}_shape_{shape}.pt"
    torch.save({
            'epoch': epochs,
            'model': model,
            'loss': loss
            }, path)
    return "Model saved"

def Batch_Data(data,batch_size,shuffle):
    #Power of 2
    """
    data : tuple ((x_start, x_end, t_start, t_end, dt, u), dxdt)
    batch_size : int
    shuffle : bool
    """
    nsamples = data[1].shape[0]

    if shuffle:
        permutation = torch.randperm(nsamples)
    else:
        permutation = torch.arange(nsamples)

    nbatches = np.ceil(nsamples/batch_size).astype(int)
    batched = [(None,None)] *nbatches  #((x_start, x_end, t_start, t_end, dt, u), dxdt)

    for i in range(nbatches):
        indices = permutation[i * batch_size : (i + 1) * batch_size]
        input_tuple = [data[0][j][indices] for j in range(len(data[0]))]
        dudt = data[1][indices]
        batched[i] = (input_tuple, dudt)

    return batched

def train_one_epoch(model,batched_train_data,loss_func,optimizer,integrator):
    computed_loss = 0.0
    optimizer.zero_grad()
    for input_tuple, dudt in batched_train_data:
      
        (u_start, u_end, t_start, t_end, dt, u_ex) = input_tuple
        n,m = u_start.shape
        #Reshaping
        if n ==1:
            u_start = u_start.view(-1)

        dudt = dudt.view(n,m)
        #Estimating dudt
        dudt_est = model.time_derivative_step(integrator = integrator, u_start = u_start,u_end = u_end,dt = dt)

        loss = loss_func(dudt_est,dudt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        computed_loss += loss.item()

    return computed_loss / len(batched_train_data)

def compute_validation_loss(model, integrator, val_data, valdata_batched, loss_func):
    val_loss = 0
    if valdata_batched is not None:
        for input_tuple, dudt in valdata_batched:
            (u_start, u_end, t_start, t_end, dt, u_ex) = input_tuple
            n,m = u_start.shape
            #Reshaping
            if n ==1:
                u_start = u_start.view(-1)
            #u_start = u_start.requires_grad_()
            dudt = dudt.view(n,m)
        
            dudt_est = model.time_derivative_step(integrator = integrator, u_start = u_start,u_end = u_end,dt = dt)

            val_loss += loss_func(dudt_est, dudt).item()
    else:
        (u_start, u_end, t_start, t_end, dt, u_ex), dudt = val_data
        n,m = u_start.shape
        #Reshaping
        if n ==1:
            u_start = u_start.view(-1)
        #u_start = u_start.requires_grad_()
        dudt = dudt.view(n,m)
        dudt_est = model.time_derivative(integrator, u_start,u_end,dt)
        val_loss = loss_func(dudt_est, dudt).item()
    val_loss = val_loss / len(valdata_batched)
    return val_loss#.item() #float(val_loss.detach().numpy())
    

def train(model,integrator, train_data,val_data, optimizer,shuffle,loss_func=torch.nn.MSELoss(),batch_size=1024,epochs = 20, verbose =True, name_sys = "Kepler"):
    """
    traindata : tuple((x_start, x_end, t_start, t_end, dt, u), dxdt)
    optimizer : torch optimizer"""
   
    trainingdetails={}
    train_batch = Batch_Data(train_data, batch_size, shuffle)
    valdata_batched = Batch_Data(val_data, batch_size, False)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    loss_list = []
    val_loss_list =  []

    print("Batch data epoch len: ", len(train_batch))
    print("Batch data shape: ", train_batch[0][0][0].shape)

    with trange(epochs) as steps:
        for epoch in steps:
            if shuffle:
                train_batch = Batch_Data(train_data,batch_size,shuffle)
            model.train(True) 
            start = datetime.datetime.now() 
            avg_loss = train_one_epoch(model,train_batch,loss_func,optimizer,integrator)
            end = datetime.datetime.now() 
            print("Training loss: ",avg_loss)
            loss_list.append(avg_loss)
            model.train(False) 
            if verbose: #Print
                steps.set_postfix(epoch=epoch, loss=avg_loss)

            if val_data is not None:
                start = datetime.datetime.now()
                vloss = compute_validation_loss(model, integrator, val_data, valdata_batched, loss_func)
                end = datetime.datetime.now()
                print("Validation loss: ",vloss)
                val_loss_list.append(vloss)

            trainingdetails["epochs"] = epoch + 1
            trainingdetails["val_loss"] = vloss
            trainingdetails["train_loss"] = avg_loss




     # Plot the loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(loss_list, label = "Training Loss")
    plt.plot(val_loss_list,label = "Validation Loss")
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    shape_data = (len(train_batch),train_batch[0][0][0].shape)

    #Saving model
    save_model(epochs, model, loss_list,model.act1,model.act2,batch_size, integrator, sys = name_sys, shape = shape_data)
    return model,trainingdetails

            