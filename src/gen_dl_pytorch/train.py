from tqdm import tqdm


def train(net, dataloader, criterion, optimizer, num_epochs):
    for epoch in tqdm(range(num_epochs)):

        running_loss = 0.0
        count = 0.0
        for data in dataloader:
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        print(f"Epoch: {epoch} Loss: {running_loss / count}")
