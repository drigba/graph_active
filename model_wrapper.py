import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

class ModelWrapper(torch.nn.Module):
    def __init__(self, model,optimizer = None, loss_fn = None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
    def train_step(self, data):
        out = self(data)
        out = F.log_softmax(out, dim=1)
        loss = self.loss_fn(out[data.train_mask], data.y[data.train_mask])
        return loss
    
    def forward(self, data):
        return self.model(data)
        
    def test_step(self, data):
        out = self(data)
        out = F.log_softmax(out, dim=1)
        return out    
    
    def predict(self,data):
        out = self.test_step(data)
        pred = out.argmax(dim=1)
        return pred
    

    
class ContrastiveModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer, loss_fn, augmentor):
        super().__init__(model,optimizer, loss_fn)
        self.predictor = LogisticRegression()
        self.augmentor = augmentor
        self.fitted = False
        
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
        out = self(data)
        out = out.detach().cpu().numpy()
        y = data.y.detach().cpu().numpy()
        train_mask = data.train_mask.detach().cpu().numpy()
        
        
        if not self.fitted:
            self.predictor.fit(out[train_mask], y[train_mask])
            self.fitted = True
            
        pred_log_probas = self.predict_log_proba(out)
        pred_log_probas = torch.tensor(pred_log_probas).to(data.y.device)
        return pred_log_probas
    
    def predict_log_proba(self,x):
        pred = self.predictor.predict_log_proba(x)
        return pred


    def reset_predictor(self):
        self.predictor = LogisticRegression()
        self.fitted = False
        
# TODO: USE ONLY GRACE LOSS AND ENCODER
class GRACEModelWrapper(ContrastiveModelWrapper):
    def __init__(self, model,optimizer):
        super().__init__(model,optimizer,None,None)

    def train_step(self, data):
        loss = self.model.train_step(data.x, data.edge_index)
        return loss
    
    def forward(self, data):
        out = self.model(data.x, data.edge_index)
        return out
    

    
