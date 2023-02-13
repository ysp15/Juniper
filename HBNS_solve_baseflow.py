import os
import sys
import json

config_path = sys.argv[1]
with open(config_path, "r") as read_file:
    data = json.load(read_file)

import deepxde as dde
from deepxde import config
from deepxde.backend import tf
import matplotlib.pyplot as plt
import numpy as np

os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/apps/cuda/11.2.2/"

if data["float"] == "64":
    dde.config.set_default_float("float64")
else:
    print("float32")

def func_zeros(X):
    return 0*X[:, 0:1]

def func_const(val):
    def output(X):
        return 0*X[:, 0:1] + val

    return output

def curl(X,Q):
    u0x = dde.gradients.jacobian(Q,X, i=0, j=0)
    u0y = dde.gradients.jacobian(Q,X, i=0, j=1)

    v0x = dde.gradients.jacobian(Q,X, i=1, j=0)
    v0y = dde.gradients.jacobian(Q,X, i=1, j=1)
    
    curl0  = v0x - u0y
    
    return [curl0]

def HBNS_0(nu=1.0):
    def pde(X,Q):
        u0 = Q[:,0:1]
        v0 = Q[:,1:2]
        p0 = Q[:,2:3]
        
        u0x = dde.gradients.jacobian(Q,X, i=0, j=0)
        u0y = dde.gradients.jacobian(Q,X, i=0, j=1)

        v0x = dde.gradients.jacobian(Q,X, i=1, j=0)
        v0y = dde.gradients.jacobian(Q,X, i=1, j=1)

        p0x = dde.gradients.jacobian(Q,X, i=2, j=0)
        p0y = dde.gradients.jacobian(Q,X, i=2, j=1)
        #########################################################
        u0xx = dde.gradients.hessian(Q,X, component=0, i=0, j=0)
        u0yy = dde.gradients.hessian(Q,X, component=0, i=1, j=1)

        v0xx = dde.gradients.hessian(Q,X, component=1, i=0, j=0)
        v0yy = dde.gradients.hessian(Q,X, component=1, i=1, j=1)
        #########################################################

        return [
            u0x + v0y,
            -nu*(u0xx + u0yy) + p0x + (u0*u0x + v0*u0y),
            -nu*(v0xx + v0yy) + p0y + (u0*v0x + v0*v0y),
            ]

    return pde

class StressBC(dde.icbc.DirichletBC):
    def __init__(
        self, geom, func, on_boundary, component=0, nu = 1.0):
        super().__init__(geom, func, on_boundary, component)
        self.nu = nu
        
    def normal_vec(self, X, beg, end):
        return self.boundary_normal(X, beg, end, None)


    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        
        ux = dde.gradients.jacobian(outputs,inputs, i=0, j=0)[beg:end]
        uy = dde.gradients.jacobian(outputs,inputs, i=0, j=1)[beg:end]
        
        vx = dde.gradients.jacobian(outputs,inputs, i=1, j=0)[beg:end]
        vy = dde.gradients.jacobian(outputs,inputs, i=1, j=1)[beg:end]

        p = outputs[beg:end, 2:3]
        
        D11 = ux
        D12 = 0.5*(uy + vx)
        D22 = vy

        sig11 = -p + 2*self.nu*D11
        sig12 = 2*self.nu*D12
        sig22 = -p + 2*self.nu*D22
        
        n = self.normal_vec(X, beg, end)
        
        a = sig11*n[:,0:1] + sig12*n[:,1:2]
        b = sig12*n[:,0:1] + sig22*n[:,1:2]

        return (a + b) - values
#################################################################
# Rectangular
Xmin, Xmax = -10.0, 30.0
Ymin, Ymax = -10.0, 10.0

Re = data['Re']
Uinf = 1.0
D = 1.0
nu = Uinf*D/Re
##############
N_colloc = data['colloc']
N_train, N_bc= N_colloc

param = data['param']
nodes, layers, lr = param

print(nodes)
print(layers)

initialiser = "Glorot uniform"

final_path = ""

path = os.getcwd()

if not os.path.isdir(path + final_path + "/Model"):
    os.makedirs(path + final_path + "/Model")

iter_list = data['iter']
iter, iter2 = iter_list
##################################################################
cylinder = dde.geometry.Disk([0, 0], D/2)
rectangle = dde.geometry.Rectangle([Xmin, Ymin], [Xmax, Ymax])

domain = rectangle - cylinder
####################################################################
def boundary_inlet(x, on_boundary):
    return on_boundary and np.isclose(x[0],Xmin)

def boundary_wall(x, on_boundary):
    return on_boundary and cylinder.on_boundary(x)

def boundary_side(x, on_boundary):
    return on_boundary and (np.isclose(x[1],Ymin) or np.isclose(x[1],Ymax))

def boundary_outlet(x, on_boundary):
    return on_boundary and np.isclose(x[0],Xmax)
####################################################################
pde = HBNS_0(nu)
##############################################################################################
bc_list = []
weights = data["w_pde"]#[1e0,1e0,1e0]

bc_inlet_u0  = dde.icbc.DirichletBC(domain, func_const(Uinf), boundary_inlet, component=0)
bc_inlet_v0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_inlet, component=1)
bc_list.append(bc_inlet_u0)
bc_list.append(bc_inlet_v0)
weights.append(data["w_in"][0])
weights.append(data["w_in"][1])

if(data['BC_in_p']):
    bc_inlet_p0  = dde.icbc.NeumannBC(domain, func_zeros, boundary_inlet, component=2)
    bc_list.append(bc_inlet_p0)
    weights.append(data["w_in"][2])

######################
bc_wall_u0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_wall, component=0)
bc_wall_v0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_wall, component=1)
bc_list.append(bc_wall_u0)
bc_list.append(bc_wall_v0)
weights.append(data["w_wall"][0])
weights.append(data["w_wall"][1])

if(data['BC_wall_p']):
    bc_wall_p0  = dde.icbc.NeumannBC(domain, func_zeros, boundary_wall, component=2)
    bc_list.append(bc_wall_p0)
    weights.append(data["w_wall"][2])

###########################
if(data['BC_side'] == 'stress'):
    bc_side_0 = StressBC(domain, func_zeros, boundary_side, component=0, nu = nu)
    bc_list.append(bc_side_0)
    weights.append(data["w_side"][0])
    
elif(data['BC_side'] == 'slip'):
    bc_side_u0  = dde.icbc.NeumannBC(domain, func_zeros, boundary_side, component=0)
    bc_side_v0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_side, component=1)
    bc_list.append(bc_side_u0)
    bc_list.append(bc_side_v0)
    weights.append(data["w_side"][0])
    weights.append(data["w_side"][1])
    
elif(data['BC_side'] == 'inlet'):
    bc_side_u0  = dde.icbc.DirichletBC(domain, func_const(Uinf), boundary_side, component=0)
    bc_side_v0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_side, component=1)
    bc_list.append(bc_side_u0)
    bc_list.append(bc_side_v0)
    weights.append(data["w_side"][0])
    weights.append(data["w_side"][1])
#########################
if(data['BC_out'] == 'stress'):
    bc_outlet_0 = StressBC(domain, func_zeros, boundary_outlet, component=0, nu = nu)
    bc_list.append(bc_outlet_0)
    weights.append(data["w_out"][0])
    
elif(data['BC_out'] == 'zeroGrad'):  
    bc_outlet_u0  = dde.icbc.NeumannBC(domain, func_zeros, boundary_outlet, component=0)
    bc_outlet_v0  = dde.icbc.NeumannBC(domain, func_zeros, boundary_outlet, component=1)
    bc_outlet_p0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_outlet, component=2)
    bc_list.append(bc_outlet_u0)
    bc_list.append(bc_outlet_v0)
    bc_list.append(bc_outlet_p0)
    weights.append(data["w_out"][0])
    weights.append(data["w_out"][1])
    weights.append(data["w_out"][2])
#######################################################################################
data = dde.data.PDE(
        domain,
        pde,
        bc_list,
        N_train,
        N_bc
        )

plt.plot(data.train_x_bc[:,0], data.train_x_bc[:,1],'rx')
plt.plot(data.train_x_all[:,0], data.train_x_all[:,1],'g.')
np.save(final_path + "DataPoints.npy", data.train_x_all)
plt.savefig(final_path + "DataPoints.png")
plt.clf()
##########################################################################
layer_size = [2] + [nodes] * layers + [3]
net = dde.nn.FNN(layer_size,"tanh",initialiser)

model = dde.Model(data,net)
###################################################
#chkpt = dde.callbacks.ModelCheckpoint(filepath, verbose=0, save_better_only=True, period=10000, monitor='train loss')
es = dde.callbacks.EarlyStopping(min_delta=0, patience=10000)
cb_list = [es]

model.compile("adam",lr=lr,loss_weights=weights)
history, train_state = model.train(iter,display_every=100, callbacks = cb_list)

dde.optimizers.config.set_LBFGS_options(maxiter=iter2)
model.compile("L-BFGS-B", loss_weights=weights)
history_2, train_state_2 = model.train()

loss_train = np.asarray(history_2.loss_train,dtype=np.float32)
loss_step = np.asarray(history_2.steps,dtype=np.float32)

np.save(final_path + "loss.npy",loss_train)
np.save(final_path + "iter.npy",loss_step)

model.save(final_path + "Model/model-last")

xt = np.linspace(Xmin,Xmax,400)
yt = np.linspace(Ymin,Ymax,200)

Xg, Yg = np.meshgrid(xt, yt, indexing='ij')
X = np.stack((Xg.flatten(),Yg.flatten()),axis=-1)

Q_pred = model.predict(X)
res = model.predict(X, operator=pde)
res = np.asarray(res,dtype=np.float32)
res = np.swapaxes(np.squeeze(res, axis=2), 0,1)

np.save(final_path + "X.npy",X)
np.save(final_path + "Q.npy",Q_pred)
np.save(final_path + "Res.npy",res)

xt = np.linspace(-2,10,600)
yt = np.linspace(-5,5,500)

Xg, Yg = np.meshgrid(xt, yt, indexing='ij')
X = np.stack((Xg.flatten(),Yg.flatten()),axis=-1)

Q_pred = model.predict(X)
res = model.predict(X, operator=pde)
res = np.asarray(res,dtype=np.float32)
res = np.swapaxes(np.squeeze(res, axis=2), 0,1)

n = 0
vlim = np.amax(np.abs(Q_pred[:,n]))
plt.scatter(X[:,0],X[:,1], c = Q_pred[:,n], vmin=-vlim, vmax=vlim)
plt.set_cmap('seismic')
plt.colorbar()
plt.savefig(final_path + "u0" + ".png")
plt.clf()

n = 1
vlim = np.amax(np.abs(Q_pred[:,n]))
plt.scatter(X[:,0],X[:,1], c = Q_pred[:,n], vmin=-vlim, vmax=vlim)
plt.set_cmap('seismic')
plt.colorbar()
plt.savefig(final_path + "v0" + ".png")
plt.clf()

n = 2
vlim = np.amax(np.abs(Q_pred[:,n]))
plt.scatter(X[:,0],X[:,1], c = Q_pred[:,n], vmin=-vlim, vmax=vlim)
plt.set_cmap('seismic')
plt.colorbar()
plt.savefig(final_path + "p0" + ".png")
plt.clf()

np.save(final_path + "Xz.npy",X)
np.save(final_path + "Qz.npy",Q_pred)
np.save(final_path + "Resz.npy",res)

Q_pred = model.predict(X, operator=curl)
np.save(final_path + "curl0.npy",Q_pred[0])

n = 0
vlim = np.amax(np.abs(Q_pred[n]))
plt.scatter(X[:,0],X[:,1], c = Q_pred[n], vmin=-vlim, vmax=vlim)
plt.set_cmap('seismic')
plt.colorbar()
plt.savefig(final_path + "curl0" + ".png")
plt.clf()

n = 0
vlim = np.amax(np.abs(res[:,n]))
plt.scatter(X[:,0],X[:,1], c = res[:,n], vmin=-vlim, vmax=vlim)
plt.set_cmap('seismic')
plt.colorbar()
plt.savefig(final_path + "mass" + ".png")
plt.clf()

n = 1
vlim = np.amax(np.abs(res[:,n]))
plt.scatter(X[:,0],X[:,1], c = res[:,n], vmin=-vlim, vmax=vlim)
plt.set_cmap('seismic')
plt.colorbar()
plt.savefig(final_path + "momX" + ".png")
plt.clf()

n = 2
vlim = np.amax(np.abs(res[:,n]))
plt.scatter(X[:,0],X[:,1], c = res[:,n], vmin=-vlim, vmax=vlim)
plt.set_cmap('seismic')
plt.colorbar()
plt.savefig(final_path + "momY" + ".png")
plt.clf()