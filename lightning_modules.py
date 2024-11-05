import lightning as L
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

class BasicLightningModule(L.LightningModule):
    def __init__(self, model, lr, loss_fn = torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr


    def training_step(self, data, batch_idx):
        train_mask = data.train_mask
        out = self.model(data)
        loss = self.loss_fn(out[train_mask], data.y[train_mask])
        return loss
    
    def test_step(self, data, batch_idx):
        out = self.model(data)
        pred = out.argmax(dim=1)
        
        test_mask = data.test_mask
        
        correct = (pred[test_mask] == data.y[test_mask]).sum()
        acc = int(correct) / int(test_mask.sum())
        
        out = torch.exp(out)
        y_true = data.y[test_mask].cpu().numpy()
        y_score = out[test_mask].cpu().detach().numpy()
        
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        self.log('test_acc', acc)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)
    
class ContrastiveLightningModule(L.LightningModule):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        
    def training_step(self, data, batch_idx):
        out = self.model(data)
        loss = self.loss_fn(out, data.y)
        return loss
    
    def test_step(self, data, batch_idx):
        out = self.model(data)
        
        out = out.detach().cpu().numpy()
        
        train_mask = data.train_mask
        test_mask = data.test_mask
        
        predictor = LogisticRegression()
        predictor.fit(out[train_mask], data.y[train_mask])
        pred = torch.tensor(predictor.predict(out[test_mask]))
            
        correct = (pred[test_mask] == data.y[test_mask]).sum()
        acc = int(correct) / int(test_mask.sum())
        
        y_true = data.y[test_mask].cpu().numpy()
        y_score = predictor.predict_proba(out[test_mask])
        
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        self.log('test_acc', acc)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)
