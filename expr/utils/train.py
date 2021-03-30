import time
import copy
import torch

def train_model(model, dataloaders, dataset_sizes, device, num_epochs=1):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Start Training the model for pre-set number of epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        model.train()

        # Iterate over data.
        for i, (inputs, labels) in enumerate(dataloaders["train"], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

            if i % 10 == 0:
                print('Batch {}/{}'.format(i + 1, len(dataloaders["train"])))

        # statistics
        epoch_loss = running_loss/dataset_sizes["train"]
        epoch_acc = running_corrects.double()/dataset_sizes["train"]

        print('Train Loss: {:.4f} Acc: {:.4f}%'.format(epoch_loss, 100*epoch_acc))    

        # exp_lr_scheduler.step()

        # Evaluate the model
        model.eval()

        running_corrects = 0
        running_total = 0

        for (inputs, labels) in dataloaders["val"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            running_total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()

        val_acc = running_corrects/running_total

        print('Valid Acc: {:.4f} %'.format(val_acc*100))    

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    end = time.time() - since

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, best_acc, end
