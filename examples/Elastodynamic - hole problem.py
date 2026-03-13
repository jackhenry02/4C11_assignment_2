
import numpy as np
import os
import time
from pyDOE import lhs
import matplotlib
import platform
if platform.system()=='Linux':
    matplotlib.use('Agg')
if platform.system()=='Windows':
    from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import pickle
import math
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

#### Creating distance Modelwith Pytorch

class ANN_DistModel(nn.Module):
    def __init__(self, N_INPUT=2, N_OUTPUT=5, N_HIDDEN=20, N_LAYERS=4):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.apply(self._init_weights)
    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.xavier_normal_(module.weight)
          if module.bias is not None:
              module.bias.data.zero_()
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


#### Creating part Modelwith Pytorch

class ANN_PartModel(nn.Module):
    def __init__(self, N_INPUT=2, N_OUTPUT=5, N_HIDDEN=20, N_LAYERS=4):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.apply(self._init_weights)
    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.xavier_normal_(module.weight)
          if module.bias is not None:
              module.bias.data.zero_()
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


#### Creating Modelwith Pytorch

class ANN_UvModel(nn.Module):
    def __init__(self, N_INPUT=2, N_OUTPUT=5, N_HIDDEN=70, N_LAYERS=8):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.apply(self._init_weights)
    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.xavier_normal_(module.weight)
          if module.bias is not None:
              module.bias.data.zero_()
                
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

def GenDistPt(xmin, xmax, ymin, ymax, xc, yc, r, num_surf_pt, num):
    # num: number per edge
    # num_t: number time step
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    x, y = np.meshgrid(x, y)
    # Delete point in hole
    dst = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5
    x = x[dst >= r]
    y = y[dst >= r]
    x = x.flatten()[:, None]
    y = y.flatten()[:, None]
    # Refinement point near hole surface
    theta = np.linspace(0.0, np.pi / 2.0, num_surf_pt)
    x_surf = np.multiply(r, np.cos(theta)) + xc
    y_surf = np.multiply(r, np.sin(theta)) + yc
    x_surf = x_surf.flatten()[:, None]
    y_surf = y_surf.flatten()[:, None]
    x = np.concatenate((x, x_surf), 0)
    y = np.concatenate((y, y_surf), 0)
    return x,y

def GenDist(XY_dist):
    dist_u = np.zeros_like(XY_dist[:, 0:1])
    dist_v = np.zeros_like(XY_dist[:, 0:1])
    dist_s11 = np.zeros_like(XY_dist[:, 0:1])
    dist_s22 = np.zeros_like(XY_dist[:, 0:1])
    dist_s12 = np.zeros_like(XY_dist[:, 0:1])
    for i in range(len(XY_dist)):
        dist_u[i, 0] = XY_dist[i][0]  # min(t, x-(-0.5))
        dist_v[i, 0] =  XY_dist[i][1]  # min(t, sqrt((x+0.5)^2+(y+0.5)^2))
        dist_s11[i, 0] = 0.5 - XY_dist[i][0]
        dist_s22[i, 0] = 0.5 - XY_dist[i][1]
        dist_s12[i, 0] = min(XY_dist[i][1], 0.5 - XY_dist[i][1], XY_dist[i][0], 0.5 - XY_dist[i][0])
    DIST = np.concatenate(( dist_u, dist_v, dist_s11, dist_s22, dist_s12), 1)
    return XY_dist , DIST

def DelHolePT(XY_c, xc=0, yc=0, r=0.1):
    # Delete points within hole
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst > r, :]

def GenHoleSurfPT(xc, yc, r, N_PT):
    # Generate
    theta = np.linspace(0.0, np.pi / 2.0, N_PT)
    xx = np.multiply(r, np.cos(theta)) + xc
    yy = np.multiply(r, np.sin(theta)) + yc
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    return xx, yy
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current cuda device: ',torch.cuda.get_device_name(0))


E = 20.0
mu = 0.25
rho = 1.0
hole_r = 0.1
#### Note: The detailed description for this case can be found in paper:
#### Physics informed deep learning for computational elastodynamicswithout labeled data.
#### https://arxiv.org/abs/2006.08472
#### But network configuration might be slightly different from what is described in paper.
PI = math.pi

# Domain bounds for x, y and t
lb = np.array([0, 0])
ub = np.array([0.5, 0.5])

# Network configuration
uv_layers   = [2] + 8 * [70] + [5]
dist_layers = [2] + 4 * [20] + [5]
part_layers = [2] + 4 * [20] + [5]

# Generate distance function for spatio-temporal space
x_dist, y_dist = GenDistPt(xmin=0, xmax=0.5, ymin=0, ymax=0.5, xc=0, yc=0, r=0.1,
                                    num_surf_pt=40, num=100)
XY_dist = np.concatenate((x_dist, y_dist), 1)
XY_dist,DIST = GenDist(XY_dist)

# Collocation point for equation residual
XY_c = lb + (ub - lb) * lhs(2, 40000)
XY_c_ref = lb + np.array([0.2, 0.2]) * lhs(2, 20000)  # Refinement for stress concentration
XY_c = np.concatenate((XY_c, XY_c_ref), 0)
XY_c = DelHolePT(XY_c, xc=0, yc=0, r=0.1)

xx, yy = GenHoleSurfPT(xc=0, yc=0, r=0.1, N_PT=5000)

HOLE = np.concatenate((xx, yy), 1)
LW = np.array([0.1, 0.0]) + np.array([0.4, 0.0]) * lhs(2, 5000)
UP = np.array([0.0, 0.5]) + np.array([0.5, 0.0]) * lhs(2, 5000)
LF = np.array([0.0, 0.1]) + np.array([0.0, 0.4]) * lhs(2, 5000)
RT = np.array([0.5, 0.0]) + np.array([0.0, 0.5]) * lhs(2, 8000)

# t_RT = RT[:, 2:3]
# period = 5  # two period in 10s
# s11_RT = 0.5 * np.sin((2 * PI / period) * t_RT + 3 * PI / 2) + 0.5
s11_RT=np.ones(RT[:,0:1].shape)
RT = np.concatenate((RT, s11_RT), 1)

# Add some boundary points into the collocation point set

XY_c = np.concatenate((XY_c, HOLE[::1, :], LF[::5, :], RT[::5, 0:2], UP[::5, :], LW[::5, :]), 0)

xx, yy = GenHoleSurfPT(xc=0, yc=0, r=0.1, N_PT=XY_c.shape[0])

HOLE = np.concatenate((xx, yy), 1)
XY_dist=torch.FloatTensor(XY_dist).requires_grad_(True)
DIST=torch.FloatTensor(DIST).requires_grad_(True)
HOLE=torch.FloatTensor(HOLE).requires_grad_(True)
LW=torch.FloatTensor(LW).requires_grad_(True)
LF=torch.FloatTensor(LF).requires_grad_(True)
UP=torch.FloatTensor(UP).requires_grad_(True)
RT=torch.FloatTensor(RT).requires_grad_(True)
XY=torch.FloatTensor(XY_c).requires_grad_(True)


model_acc_dist=100000
model_acc_part=10000
model_acc_uv=100000

# train standard neural network to fit distance training data

torch.manual_seed(123)
model_dist = ANN_DistModel()
model_dist.load_state_dict(torch.load('weights/distance_hole.pth'))
model_dist.to(device)
optimizer_dist = torch.optim.Adam(model_dist.parameters(),lr=1e-3)
for i in range(1000):
    optimizer_dist.zero_grad()
    XY_dist=XY_dist.to(device)
    DIST=DIST.to(device)
    # compute the "data loss"
    yh = model_dist(XY_dist)

    loss1 = torch.mean((yh-DIST)**2)# use mean squared error
  
    
    # backpropagate joint loss
    # loss=loss1
    loss = 1000*(loss1)# add two loss terms together
    if i%100==0: 
        if model_acc_dist>loss.item():
            model_acc_dist=loss.item()
            torch.save(model_dist.state_dict(), "weights/distance_hole.pth")
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))

    loss.backward()
    optimizer_dist.step()
    
print(model_acc_dist)
model_dist = ANN_DistModel()
model_dist.load_state_dict(torch.load('weights/distance_hole.pth'))
model_dist.to(device)

# torch.save(model_dist.state_dict(), "./distance_paper.pth")
# train standard neural network to fit distance training data

torch.manual_seed(123)
model_part = ANN_PartModel()
model_part.load_state_dict(torch.load('weights/part_hole.pth'))
model_part.to(device)
optimizer_part = torch.optim.Adam(model_part.parameters(),lr=1e-3)
for i in range(1000):
    optimizer_part.zero_grad()

    #LF
    LF=LF.to(device);
    yh_lf = model_part(LF)
    loss1=torch.mean((yh_lf[:,[0,4]])**2);

    #UP
    UP=UP.to(device);
    yh_up = model_part(UP)
    loss1=loss1+torch.mean((yh_up[:,[3,4]])**2);
    #LW
    LW=LW.to(device);
    yh_lw = model_part(LW)
    loss1=loss1+torch.mean((yh_lw[:,[1,4]])**2);

    #RT
    RT=RT.to(device);
    yh_rt = model_part(RT[:,0:2])
    loss1=loss1+torch.mean((yh_rt[:,4])**2);
    loss1=loss1+torch.mean((yh_rt[:,2]-RT[:,2])**2);
    # backpropagate joint loss
    # loss=loss1
    loss = 1000*(loss1)# add two loss terms together
    if i%100==0: 
        if model_acc_part>loss.item():
            model_acc_part=loss.item()
            torch.save(model_part.state_dict(), "weights/part_hole.pth")
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))

    loss.backward()
    optimizer_part.step()


print(model_acc_part)
model_part = ANN_PartModel()
model_part.load_state_dict(torch.load('weights/part_hole.pth'))
model_part.to(device)
# torch.save(model_part.state_dict(), "./part_paper.pth")
# train standard neural network to fit distance training data

torch.manual_seed(123)
model_uv = ANN_UvModel()
if os.path.isfile('weights/uv_paper.pth'):
    model_uv.load_state_dict(torch.load('weights/uv_paper.pth'))
    print("loaded uv model successfully...")

model_uv.to(device)


class Build_Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = XY
        self.y = HOLE
        self.len = self.x.shape[0]
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len
train = Build_Data()
train_loader = DataLoader(train, batch_size=128,shuffle=True)
optimizer_uv = torch.optim.Adam(model_uv.parameters(),lr=1e-4)
for i in range(1):
    
    running_loss=0.0
    loss_hole=0.0
    loss_plate_stress=0.0 
    loss_plate_uv=0.0
    for _, [xy,hole] in enumerate(train_loader):

        optimizer_uv.zero_grad()
    

        xy=xy.to(device)
        yh = model_uv(xy)
        yh=yh*model_dist(xy)+model_part(xy)

        e11=torch.autograd.grad(yh[:,0], xy,torch.ones_like(yh[:,0]), create_graph=True)[0][:,0]# computes dy/dx
        e12=torch.autograd.grad(yh[:,0], xy,torch.ones_like(yh[:,0]), create_graph=True)[0][:,1]
        e22=torch.autograd.grad(yh[:,1], xy,torch.ones_like(yh[:,1]), create_graph=True)[0][:,1]# computes dy/dx
        e12=e12+torch.autograd.grad(yh[:,1], xy,torch.ones_like(yh[:,1]), create_graph=True)[0][:,0]
        sp11 = (E * e11)/ (1 - mu * mu) + (E * mu * e22) / (1 - mu * mu)
        sp22 = (E * mu* e11) / (1 - mu * mu)  + (E * e22) / (1 - mu * mu) 
        sp12 = (E* e12) / (2 * (1 + mu)) 

        f_s11 = yh[:,2] - sp11
        f_s12 = yh[:,4] - sp12
        f_s22 = yh[:,3] - sp22

        s11_1=torch.autograd.grad(yh[:,2], xy,torch.ones_like(yh[:,2]), create_graph=True)[0][:,0]
        s12_2=torch.autograd.grad(yh[:,4], xy,torch.ones_like(yh[:,4]), create_graph=True)[0][:,1]

        s22_2=torch.autograd.grad(yh[:,3], xy,torch.ones_like(yh[:,3]), create_graph=True)[0][:,1]
        s12_1=torch.autograd.grad(yh[:,4], xy,torch.ones_like(yh[:,4]), create_graph=True)[0][:,0]
 


        f_u = s11_1 + s12_2 
        f_v = s22_2 + s12_1 


        loss1 = torch.mean(f_s11**2)+torch.mean(f_s22**2)+torch.mean(f_s12**2)# use mean squared error
        loss3=torch.mean(f_u**2)+torch.mean(f_v**2)
    ## For hole 

        hole=hole.to(device)
        r = hole_r
        nx = -hole[:,0] / r
        ny = -hole[:,1] / r
        yh_hole = model_uv(hole)
        yh_hole=yh_hole*model_dist(hole)+model_part(hole)
        tx=torch.mul(yh_hole[:,2],nx)+torch.mul(yh_hole[:,4],ny)
        ty=torch.mul(yh_hole[:,4],nx)+torch.mul(yh_hole[:,3],ny)
    # tx=yh[:,2]*nx+yh[:,4]*ny
    # ty=yh[:,4]*nx+yh[:,3]*ny
        loss2=torch.mean(tx**2)+torch.mean(ty**2)
    
    # backpropagate joint loss
    # loss=loss1
        loss = loss1 + loss2+loss3# add two loss terms together 
        loss=10*loss
        loss.backward()
        optimizer_uv.step()
        running_loss+=loss.item()
        loss_hole+=loss2.item()
        loss_plate_stress+=loss1.item()
        loss_plate_uv+=loss3.item()
        if _%100==99: 
            print("palte loss stress: ",loss_plate_stress/100,"palte loss uv: ",loss_plate_uv/100,"  hole loss: ", loss_hole/100)
            torch.save(model_uv.state_dict(), "weights/uv_paper.pth")
            print('[%d, %5d] loss: %.4f' %
                (i + 1, _ + 1, running_loss / 100))
            running_loss = 0.0
            loss_hole = 0.0
            loss_plate_stress = 0.0
            loss_plate_uv=0.0


    
# torch.save(model_uv.state_dict(), "./uv_paper.pth")



def postProcessDef(xmin, xmax, ymin, ymax, field, s=5, num=0, scale=1):
    ''' Plot deformed plate (set scale=0 want to plot undeformed contours)
    '''
    # [x_pred, y_pred, u_pred, v_pred, s11_pred, s22_pred, s12_pred] = field
    x_pred=field[:,0]
    y_pred=field[:,1]
    u_pred=field[:,2]
    v_pred=field[:,3]
    s11_pred=field[:,4]
    s22_pred=field[:,5]
    s12_pred=field[:,6]
    print(v_pred)
    #
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    cf = ax[0].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=u_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[0].axis('square')
    for key, spine in ax[0].spines.items():
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
            spine.set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0].set_title(r'$u$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0])
    cbar.ax.tick_params(labelsize=14)
    #
    cf = ax[1].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=v_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    for key, spine in ax[1].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[1].axis('square')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1].set_title(r'$v$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1])
    cbar.ax.tick_params(labelsize=14)
    #
    plt.savefig('./output/uv_comparison_paper' + str(num) + '.png', dpi=200)
    plt.close('all')
    #
    # Plot predicted stress
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.3)
    #
    cf = ax[0].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s11_pred, alpha=0.7, edgecolors='none',
                          marker='s', cmap='rainbow', s=s)
    ax[0].axis('square')
    for key, spine in ax[0].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0].set_title(r'$\sigma_{11}$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0])
    cbar.ax.tick_params(labelsize=14)
    
    #
    cf = ax[1].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s22_pred, alpha=0.7, edgecolors='none',
                          marker='s', cmap='rainbow', s=s)
    ax[1].axis('square')
    for key, spine in ax[1].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1].set_title(r'$\sigma_{22}$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1])
    cbar.ax.tick_params(labelsize=14)
    #
    cf = ax[2].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s12_pred, alpha=0.7, edgecolors='none',
                          marker='s', cmap='rainbow', s=s)
    ax[2].axis('square')
    for key, spine in ax[2].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xlim([xmin, xmax])
    ax[2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2].set_title(r'$\sigma_{12}$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2])
    cbar.ax.tick_params(labelsize=14)
    #
    plt.savefig('./output/stress_comparison_paper' + str(num) + '.png', dpi=200)
    plt.close('all')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

shutil.rmtree('./output', ignore_errors=True)
if not os.path.isdir('./output'):
    os.makedirs('./output')

# os.makedirs('./output')
path_fem = './platehole_data.csv'

my_data = np.genfromtxt(path_fem, delimiter=',')
csv_data=my_data[1:,:]
x_col, y_col = 2,3
u_col, v_col = 0,1
s11_col, s12_col, s22_col=5,6,8
xyt=torch.FloatTensor(csv_data[:,[x_col,y_col]]).requires_grad_(True)
xyt=xyt.to(device)
uv_fem=torch.FloatTensor(csv_data[:,[u_col,v_col]]).requires_grad_(True)
s_fem=torch.FloatTensor(csv_data[:,[s11_col,s12_col,s22_col]]).requires_grad_(True)

y=model_uv(xyt)
yd=model_dist(xyt)
yp=model_part(xyt)
y=y*yd+yp
y=y.cpu().detach().numpy()
xyt=xyt.cpu().detach().numpy()
uv_fem=uv_fem.cpu().detach().numpy()
s_fem=s_fem.cpu().detach().numpy()










x_star = np.linspace(0, 0.5, 251)
y_star = np.linspace(0, 0.5, 251)
x_star, y_star = np.meshgrid(x_star, y_star)
x_star = x_star.flatten()[:, None]
y_star = y_star.flatten()[:, None]
dst = ((x_star - 0) ** 2 + (y_star - 0) ** 2) ** 0.5
x_star = x_star[dst >= 0.1]
y_star = y_star[dst >= 0.1]
x_star = x_star.flatten()[:, None]
y_star = y_star.flatten()[:, None]
shutil.rmtree('./output', ignore_errors=True)
os.makedirs('./output')

xy=np.concatenate((x_star,y_star),1)
xy=torch.FloatTensor(xy).requires_grad_(True)
xy=xy.to(device)
y=model_uv(xy)
yd=model_dist(xy)
yp=model_part(xy)
y=y*yd+yp
xy=xy.cpu().detach().numpy()
y=y.cpu().detach().numpy()
field=np.concatenate((xy,y),1)
# print(field[0])
postProcessDef(xmin=0, xmax=0.50, ymin=0, ymax=0.50, s=4, scale=0, field=field)

# newPostProcess(field)