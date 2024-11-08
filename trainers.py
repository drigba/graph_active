class Trainer():
    def train(self,model, data, epochs):
        model.train()
        for _ in range(epochs):
            model.optimizer.zero_grad()
            loss = model.train_step(data)
            loss.backward()
            model.optimizer.step()
            print(loss.item())  
    
    def test(self,model, data):
        model.eval()
        out = model.test_step(data)
        pred = out.argmax(dim=1)
        acc = self.calculate_accuracy(pred, data)
        return acc
    
    def calculate_accuracy(self, pred, data):
        return (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    
    




