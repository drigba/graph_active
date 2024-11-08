import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

class ModelWrapper(torch.nn.Module):
    def __init__(self, model,optimizer, loss_fn):
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
        out = out.detach().cpu().numpy()
        data = data.detach().cpu().numpy()
        
        if not check_is_fitted(self.predictor):
            self.predictor.fit(out[data.train_mask], data.y[data.train_mask])
            
        pred_log_probas = torch.tensor(self.predictor.predict_log_proba(out)).to(data.device)
        return pred_log_probas
    

    def reset_predictor(self):
        self.predictor = LogisticRegression()