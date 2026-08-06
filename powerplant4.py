import torch.optim as optim

model=ANN()

criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters())
'''#train the ann
train_losses=[]
val_losses=[]


epochs=100

for epoch in range(epochs):
    model.train()
    running_loss=0.0

    for xb,yb in train_loader:
        #xb=features of 1 batch
        #yb features of second batch
        optimizer.zero_grad()

        outputs = model(xb)
        loss=crietrion(outputs,yb)
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()

    epoch_train_loss=running_loss/len(train_loader)
    train_losses.append(epoch_train_loss)
    model.eval()
    running_val_loss=0.0

    with torch.no_grad():
        for xb,yb in test_loader:
            outputs=model(xb)
            loss=crietrion(outputs,yb)
            running_val_loss+=loss

    epoch_val_loss=running_val_loss/len(test_loader)
    val_losses.append(epoch_val_loss)        

    print(f"epoch ${epoch+1}/{epochs} ==> train loss =${epoch_train_loss:.4f}& val loss=${epoch_val_loss:.4f}")'''
train_losses = []
val_losses = []
epochs = 100

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            running_val_loss += loss.item()
    epoch_val_loss = running_val_loss / len(test_loader)
    val_losses.append(epoch_val_loss)

    print(f"epoch {epoch+1}/{epochs} ==> train loss={epoch_train_loss:.4f} & val loss={epoch_val_loss:.4f}")
