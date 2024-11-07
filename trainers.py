
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


class BaseTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def train(self, data, epochs):
        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            loss = self.model.train_step(data)
            loss.backward()
            self.optimizer.step()
    
    def test(self, data):
        self.model.eval()
        out = self.model.test_step(data)
        pred = out.argmax(dim=1)
        acc = self.calculate_accuracy(pred, data)
        return acc
    
    def calculate_accuracy(self, pred, data):
        return (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    
class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn):
        super(Trainer, self).__init__(model, optimizer, loss_fn)
        
    
    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data)
        out = F.log_softmax(out, dim=1)
        loss = self.loss_fn(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def test(self, data):
        self.model.eval()
        out = self.model(data)
        out = F.log_softmax(out, dim=1)
        pred = out.argmax(dim=1)
        
        test_mask = data.test_mask
        
        correct = (pred[test_mask] == data.y[test_mask]).sum()
        acc = int(correct) / int(test_mask.sum())
        
        out = torch.exp(out)
        y_true = data.y[test_mask].cpu().numpy()
        y_score = out[test_mask].cpu().detach().numpy()
        
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        return acc, auc
    
class ContrastiveTrainer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn, augmentor):
        super().__init__(model, optimizer, loss_fn)
        self.augmentor = augmentor
    
    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        data_tmp1 = data.clone()
        data_tmp2 = data.clone()

        data_tmp1 = self.augmentor(data_tmp1)
        data_tmp2 = self.augmentor(data_tmp2)

        out1 = self.model(data_tmp1)
        out2 = self.model(data_tmp2)

        loss = self.loss_fn(out1, out2)

        loss.backward()

        self.optimizer.step()

        print(f"Loss: {loss.item()}")
        return loss.item()
    
    def test(self, data):
        self.model.eval()
        out = self.model(data)
        out = F.log_softmax(out, dim=1)

        out = out.detach().cpu().numpy()
        train_mask = data.train_mask.cpu().numpy()
        test_mask = data.test_mask.cpu().numpy()
        
        labels = data.y.cpu().numpy()
        
        predictor = LogisticRegression()
        predictor.fit(out[train_mask], labels[train_mask])
        pred = torch.tensor(predictor.predict(out))
            
        correct = (pred[test_mask] == labels[test_mask]).sum()
        acc = int(correct) / int(test_mask.sum())
        
        y_true = labels[test_mask]
        y_score = predictor.predict_proba(out[test_mask])
        
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        return acc, auc
    
    
    
class ModelWrapper():
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        
    def train_step(self, data):
        out = self.model(data)
        out = F.log_softmax(out, dim=1)
        loss = self.loss_fn(out[data.train_mask], data.y[data.train_mask])
        return loss
        
    def test_step(self, data):
        out = self.model(data)
        out = F.log_softmax(out, dim=1)
        return out    
    
    def predict(self,data):
        out = self.test_step(data)
        pred = out.argmax(dim=1)
        return pred
    
class ContrastiveModelWrapper(ModelWrapper):
    def __init__(self, model, loss_fn, augmentor):
        super().__init__(model, loss_fn)
        self.predictor = LogisticRegression()
        self.augmentor = augmentor
        
    def train_step(self, data):
        data_tmp1 = data.clone()
        data_tmp2 = data.clone()

        data_tmp1 = self.augmentor(data_tmp1)
        data_tmp2 = self.augmentor(data_tmp2)

        out1 = self.model(data_tmp1)
        out2 =self.model(data_tmp2)

        loss = self.loss_fn(out1, out2)
        return loss
    
    def test_step(self, data):
        out = self.model(data)
        if not check_is_fitted(self.predictor):
            out = out.detach().cpu().numpy()
            data = data.detach().cpu().numpy()
            self.predictor.fit(out[data.train_mask], data.y[data.train_mask])
        pred_log_probas = torch.tensor(self.predictor.predict_log_proba(out)).to(data.device)
        return pred_log_probas
    
    def reset_predictor(self):
        self.predictor = LogisticRegression()



