import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from util import *
from sklearn.cluster import KMeans as KM
from sklearn.preprocessing import LabelBinarizer

class ModelWrapper(torch.nn.Module):
    def __init__(self, model,optimizer = None, loss_fn = None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
    def train_step(self, data):
        out = self(data)
        out = F.log_softmax(out, dim=1)
        loss = self.loss_fn(out[data.train_mask], data.y_train[data.train_mask])
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
    
    def eval(self):
        self.model.eval()
        

    
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
        
        if not self.fitted:
            self._fit_predictor(data,out)
            
        pred_log_probas = self.predict_log_proba(out)
        pred_log_probas = torch.tensor(pred_log_probas).to(data.y.device)
        return pred_log_probas
    
    def predict_log_proba(self,x):
        pred = self.predictor.predict_log_proba(x)
        return pred
    
    def _fit_predictor(self,data,out):
        y = data.y.detach().cpu().numpy()
        train_mask = data.train_mask.detach().cpu().numpy()
        self.predictor.fit(out[train_mask], y[train_mask])
        self.fitted = True

    def reset_predictor(self):
        self.predictor = LogisticRegression(max_iter=1000)
        self.fitted = False
        
# TODO: USE ONLY GRACE LOSS AND ENCODER
class GRACEModelWrapper(ContrastiveModelWrapper):
    def __init__(self, model,optimizer):
        super().__init__(model,optimizer,None,None)

    def train_step(self, data):
        loss = self.model.train_step(data)
        return loss
    
class GRACEModelWrapperCluster(GRACEModelWrapper):
    
    def test_step(self, data):
        out = self(data)
        out = out.detach().cpu().numpy()
        
        if not self.fitted:
            self._fit_predictor(data,out)
            
        features = np.hstack([out,self.distances_ ,self.cluster_labels_])    
        pred_log_probas = self.predict_log_proba(features)
        pred_log_probas = torch.tensor(pred_log_probas).to(data.y.device)
        return pred_log_probas
    
    def _fit_predictor(self,data,out):
        
        y = data.y.detach().cpu().numpy()
        train_mask = data.train_mask.detach().cpu().numpy()
        
        cc = init_kmeans(data,data.train_mask,out)
        self.km = KM(n_clusters=len(np.unique(y)),init=cc,max_iter=500,tol=1e-4,random_state=0)
        self.km.fit(out)
        self.distances_ = self.km.transform(out)
        
        lb = LabelBinarizer()
        lb.fit(self.km.labels_)
        self.cluster_labels_ =  lb.transform(self.km.labels_)

        features = np.hstack([out,self.distances_ ,self.cluster_labels_])
        self.predictor.fit(features[train_mask], y[train_mask])
        self.fitted = True
            
    

    
    
class SemiSupervisedModelWrapper(torch.nn.Module):
    def __init__(self, supervised_model, self_supervised_model,optimizer, alpha = 1.0):
        super().__init__()
        self.supervised_model = supervised_model
        self.self_supervised_model = self_supervised_model
        self.supervised_loss = F.nll_loss
        self.alpha = alpha
        self.optimizer = optimizer  
        
    def train_step(self, data):
        self_supervised_loss = self.self_supervised_model.train_step(data)
        
        sup_out = self(data)
        sup_out = F.log_softmax(sup_out, dim=1)
        supervised_loss = self.supervised_loss(sup_out[data.train_mask], data.y_train[data.train_mask])
        
        loss = (1-self.alpha)*supervised_loss + self.alpha * self_supervised_loss
        return loss
    
    def forward(self, data):
        latent = self.self_supervised_model(data.x, data.edge_index)
        sup_out = self.supervised_model(latent)
        return sup_out
        
    def test_step(self, data):
        out = self(data)
        out = F.log_softmax(out, dim=1)
        return out    
    
    def predict(self,data):
        out = self.test_step(data)
        pred = out.argmax(dim=1)
        return pred
    
    def eval(self):
        self.supervised_model.eval()
        self.self_supervised_model.eval()

    
