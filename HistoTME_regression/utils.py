import math
import torch

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Macenko_Normalizer:

    def __init__(self,alpha=0.01,beta=0.15):
        self.alpha = alpha
        self.beta = beta
        
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.maxC_target = None
        #self.stain_matrix_target_RGB = None

    def get_stain_matrix(self, img):
        # img must be a torch tensor with shape (B,3,H,W), and with values in [0,1]
        B,C,H,W = img.shape
        angular_percentile = self.alpha
        
        # In TIAToolbox, they compute a tissue mask, here we ignore this step and threshold out whitespace in OD
        # Thresholding is also used in the original paper
        img_flatten = torch.flatten(img, start_dim=2, end_dim=3) # shape: (B,3,H*W)
        # convert rgb to od
        img_flatten[img_flatten==0] = 1e-3
        img_od = -torch.log(img_flatten) # shape: (B,3,H*W)
        
        mask = torch.all(img_od>self.beta,dim=1,keepdims=True) # shape: (B,1,H*W)
        img_od_selected = mask*img_od # shape: (B,3,H*W), pixel vectors with any of its components <= beta are set to 0
        
        # Need to compute convariance of nonzero vectors, this is done by np.cov in TIAToolbox,
        # but this function does not work for batched data
        img_od_mean = img_od_selected.sum(dim=2,keepdims=True)/mask.sum(dim=2,keepdims=True) # shape: (B,3,1)
        shifted = (img_od_selected - img_od_mean)*mask # shape: (B,3,H*W)
        batch_cov = shifted@shifted.transpose(1,2)/(mask.sum(dim=2,keepdims=True)-1) # shape: (B,3,3)
        
        _, eigen_vectors = torch.linalg.eigh(batch_cov) # shape: (B,3,3)
        eigen_vectors = eigen_vectors[:,:,[2,1]] # two principal components, shape: (B,3,2)
        
        eigen_vectors = eigen_vectors*torch.sign(eigen_vectors[:,0:1,:]) # point to the positive direction of x-axis
        
        proj = eigen_vectors.transpose(1,2)@img_od_selected # shape: (B,2,H*W)
        
        phi = torch.arctan2(proj[:,1,:], proj[:,0,:]) # angles, shape: (B,H*W)
        min_phi = torch.quantile(phi,q=angular_percentile,dim=1,keepdims=True) # shape: (B,1)
        max_phi = torch.quantile(phi,q=1-angular_percentile,dim=1,keepdims=True) # shape: (B,1)
        
        V1 = eigen_vectors@torch.hstack((torch.cos(min_phi), torch.sin(min_phi))).reshape((B,2,1)) # shape: (B,3,1)
        V2 = eigen_vectors@torch.hstack((torch.cos(max_phi), torch.sin(max_phi))).reshape((B,2,1)) # shape: (B,3,1)
        
        # right order of V1 and V2 
        VV1 = V1*(V1[:,0:1]>V2[:,0:1]) + V2*(V2[:,0:1]>V1[:,0:1]) # shape: (B,3,1)
        VV2 = V1*(V1[:,0:1]<=V2[:,0:1]) + V2*(V2[:,0:1]<=V1[:,0:1]) # shape: (B,3,1)
        HE = torch.cat((VV1,VV2),dim=2) # shape: (B,3,2)
        
        return HE/torch.linalg.norm(HE,dim=1,keepdims=True)

    def get_concentrations(self, img, stain_matrix):
        # img must be a torch tensor with shape (B,3,H,W), and with values in [0,1]
        # stain_matrix must be a torch tensor of shape (B,3,2)
        
        #B,C,H,W = img.shape
        img_flatten = torch.flatten(img, start_dim=2, end_dim=3) # shape: (B,3,H*W)
        
        # convert rgb to od
        img_flatten[img_flatten==0] = 1e-3
        img_od = -torch.log(img_flatten)
        img_od[img_od<1e-6] = 1e-6 # shape: (B,3,H*W)

        # get concentrations
        X = torch.linalg.pinv(stain_matrix) @ img_od # shape: (B,2,H*W)
        return X

    def fit(self, target):
        # target shape: (1,3,H,W), only 1 target image
        self.stain_matrix_target = self.get_stain_matrix(target) # shape: (1,3,2)
        self.target_concentrations = self.get_concentrations(target,self.stain_matrix_target) # shape: (1,2,H*W)
        self.maxC_target = torch.quantile(self.target_concentrations,1-self.alpha,dim=2) # shape: (1,2)
        #self.maxC_target = torch.reshape(self.maxC_target,(1,1,2))
        #self.stain_matrix_target_RGB = 

    def transform(self, img):
        # img must be a torch tensor with shape (B,3,H,W), and with values in [0,1]
        B,C,H,W = img.shape
        stain_matrix_source = self.get_stain_matrix(img) # shape: (B,3,2)
        source_concentrations = self.get_concentrations(img, stain_matrix_source) # shape: (B,2,H*W)
        max_c_source = torch.quantile(source_concentrations,1-self.alpha,dim=2) # shape: (B,2)
        source_concentrations = source_concentrations*torch.reshape(self.maxC_target/max_c_source,(B,2,1)) # shape: (B,2,H*W)
        transformed = torch.exp(-self.stain_matrix_target@source_concentrations) # shape: (B,3,H*W)
        transformed[transformed>1.0] = 1.0
        transformed[transformed<0.0] = 0.0
        return torch.reshape(transformed, (B,C,H,W))
    