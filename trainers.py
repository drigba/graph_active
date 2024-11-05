
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
class BaseTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def train(self, data, epochs):
        for _ in range(epochs):
            self.train_step(data)
    
    def train_step(self, data):
        raise NotImplementedError
    
    def test(self, data):
        raise NotImplementedError
    
class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn):
        super(Trainer, self).__init__(model, optimizer, loss_fn)
        
    
    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data)
        loss = self.loss_fn(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def test(self, data):
        self.model.eval()
        out = self.model(data)
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
    def __init__(self, model, optimizer, loss_fn, predictor):
        super().__init__(model, optimizer, loss_fn)
    
    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data)
        loss = self.loss_fn(out, data.y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def test(self, data):
        self.model.eval()
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
        return acc, auc

# Training function
def train(data,train_mask,model,optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = model.loss(out, data.y, train_mask)
    loss.backward()
    optimizer.step()
    return loss.item()

# Test function
def test(data,test_mask,model):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    
    correct = (pred[test_mask] == data.y[test_mask]).sum()
    acc = int(correct) / int(test_mask.sum())
    
    
    out =  torch.exp(out)
    y_true = data.y[test_mask].cpu().numpy()
    y_score = out[test_mask].cpu().detach().numpy()
   
    auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    return acc, auc


def nt_xent_loss(x1,x2,t=1.0):
    D, s_12, s_11, s_22 = sim_vectors(x1,x2,t)
    
    l1 = D / (D + s_12 + s_11)
    loss1 = -torch.log(l1)
    
    l2 = D / (D + s_12 + s_22)
    loss2 = -torch.log(l2)
    
    loss = (loss1 + loss2) / (2*len(x1))
    return loss

def sim_vectors(x1,x2, t=1.0):
    sims_12 = tempered_cosine_similarity(x1,x2,t)
    sims_11 = tempered_cosine_similarity(x1,x1,t)
    sims_22 = tempered_cosine_similarity(x2,x2,t)
    
    D = torch.diag(sims_12,0)
    s_12 = sims_12.sum(dim=1) - D
    s_11 = sims_11.sum(dim=1) - torch.diag(sims_11,0)
    s_22 = sims_22.sum(dim=1) - torch.diag(sims_22,0)
    
    return D, s_12, s_11, s_22


    

def tempered_cosine_similarity(x1,x2, t=1.0):
    cos_sim = F.cosine_similarity(x1[None,:,:], x2[:,None,:], dim=-1)
    cos_sim = torch.exp(cos_sim / t)
    return cos_sim