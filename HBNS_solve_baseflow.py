import os
import sys
import json

config_path = sys.argv[1]
with open(config_path, "r") as read_file:
    testcfg = json.load(read_file)

import deepxde as dde
from deepxde import config
from deepxde.backend import tf
import matplotlib.pyplot as plt
import numpy as np

os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/apps/cuda/11.2.2/"

if testcfg["float"] == "64":
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
testcfg["Xlim"][0]
Xmin, Xmax = testcfg["Xlim"][0], testcfg["Xlim"][1]
Ymin, Ymax = testcfg["Ylim"][0], testcfg["Ylim"][1]

Re = testcfg['Re']
Uinf = 1.0
D = 1.0
nu = Uinf*D/Re
##############
N_colloc = testcfg['colloc']
N_train, N_bc= N_colloc

param = testcfg['param']
nodes, layers, lr = param

print(nodes)
print(layers)

initialiser = "Glorot uniform"

final_path = ""

path = os.getcwd()

if not os.path.isdir(path + final_path + "/Model"):
    os.makedirs(path + final_path + "/Model")

iter_list = testcfg['iter']
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
weights = testcfg["w_pde"]#[1e0,1e0,1e0]

bc_inlet_u0  = dde.icbc.DirichletBC(domain, func_const(Uinf), boundary_inlet, component=0)
bc_inlet_v0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_inlet, component=1)
bc_list.append(bc_inlet_u0)
bc_list.append(bc_inlet_v0)
weights.append(testcfg["w_in"][0])
weights.append(testcfg["w_in"][1])

if(testcfg['BC_in_p']):
    bc_inlet_p0  = dde.icbc.NeumannBC(domain, func_zeros, boundary_inlet, component=2)
    bc_list.append(bc_inlet_p0)
    weights.append(testcfg["w_in"][2])

######################
bc_wall_u0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_wall, component=0)
bc_wall_v0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_wall, component=1)
bc_list.append(bc_wall_u0)
bc_list.append(bc_wall_v0)
weights.append(testcfg["w_wall"][0])
weights.append(testcfg["w_wall"][1])

if(testcfg['BC_wall_p']):
    bc_wall_p0  = dde.icbc.NeumannBC(domain, func_zeros, boundary_wall, component=2)
    bc_list.append(bc_wall_p0)
    weights.append(testcfg["w_wall"][2])

###########################
if(testcfg['BC_side'] == 'stress'):
    bc_side_0 = StressBC(domain, func_zeros, boundary_side, component=0, nu = nu)
    bc_list.append(bc_side_0)
    weights.append(testcfg["w_side"][0])
    
elif(testcfg['BC_side'] == 'slip'):
    bc_side_u0  = dde.icbc.NeumannBC(domain, func_zeros, boundary_side, component=0)
    bc_side_v0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_side, component=1)
    bc_list.append(bc_side_u0)
    bc_list.append(bc_side_v0)
    weights.append(testcfg["w_side"][0])
    weights.append(testcfg["w_side"][1])
    
elif(testcfg['BC_side'] == 'inlet'):
    bc_side_u0  = dde.icbc.DirichletBC(domain, func_const(Uinf), boundary_side, component=0)
    bc_side_v0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_side, component=1)
    bc_list.append(bc_side_u0)
    bc_list.append(bc_side_v0)
    weights.append(testcfg["w_side"][0])
    weights.append(testcfg["w_side"][1])
#########################
if(testcfg['BC_out'] == 'stress'):
    bc_outlet_0 = StressBC(domain, func_zeros, boundary_outlet, component=0, nu = nu)
    bc_list.append(bc_outlet_0)
    weights.append(testcfg["w_out"][0])
    
elif(testcfg['BC_out'] == 'zeroGrad'):  
    bc_outlet_u0  = dde.icbc.NeumannBC(domain, func_zeros, boundary_outlet, component=0)
    bc_outlet_v0  = dde.icbc.NeumannBC(domain, func_zeros, boundary_outlet, component=1)
    bc_outlet_p0  = dde.icbc.DirichletBC(domain, func_zeros, boundary_outlet, component=2)
    bc_list.append(bc_outlet_u0)
    bc_list.append(bc_outlet_v0)
    bc_list.append(bc_outlet_p0)
    weights.append(testcfg["w_out"][0])
    weights.append(testcfg["w_out"][1])
    weights.append(testcfg["w_out"][2])
#######################################################################################
if (testcfg['mesh_ref']):
    if (testcfg['dist'] = 'centre_bias'):
        
        nsamp = 1000
        a = testcfg['c_bias'][0]
        x0 = testcfg['c_bias'][1]
        b = -1/(x0+a)
        points = []
        ##################################################################
        dom_coords = np.array([[Xmin, Ymin],
                                [Xmax, Ymax]])
                                
        while(len(points) < N_train):
            x_test = dom_coords[0:1, :] + (dom_coords[1:2, :] - dom_coords[0:1, :]) * np.random.rand(nsamp, 2)
            dist = np.random.rand(nsamp, 1)
            t_b = np.random.rand(nsamp, 1)*2 - 1
        
            y0 = np.abs(x_test[:,1])
            ratY = 1 - (np.log(dist+a) + b*dist - np.log(a))
            ratY[:,0] *= y0
        
            IXt = (t_b > 0)
            IXb = (t_b <= 0)
                        
            x_test[IXt[:,0],1] = ratY[IXt[:,0],0]
            x_test[IXb[:,0],1] = - ratY[IXb[:,0],0]
        
            for i in range(x_test.shape[0]):
                if(len(points) < N_train):
                    if (x_test[i,0]**2 + x_test[i,1]**2 > (D*D/4) ):
                        if (x_test[i,1]**2 < Ymax*Ymax):
                            points.append(x_test[i,:])
        
                else:
                    break

        points = np.asarray(points)

    elif (testcfg['dist'] = 'box_ref'):
        nsamp = 1000
        
        base_size = 10.0
        wake_size = 0.1
        near_size = 0.01
        far_size  = 0.2
        
        rho_base = 4/(base_size*base_size)
        rho_wake = 4/(wake_size*wake_size)
        rho_near = 4/(near_size*near_size)
        rho_far  = 4/(far_size*far_size)
        
        Xb = -7.5
        Yb =  9.0
        A_base = 2*(Xb-Xmin)*(Yb) + 2*(Xmax - Xb)*(Ymax-Yb)
        
        Xf1 = -2.5
        Xf2 = 30.0
        Yf  = 2.5
        A_far = 2*(Xf1 - Xb)*Yf + 2*(Xf2 - Xf1)*(Yb - Yf) + 2*(Xmax - Xf2)*Yb
        
        Xw1 = -1.0
        Xw2 = 1.0
        Yw = 1.0
        A_wake = 2*(Xw1 - Xf1)*Yf + 2*(Xw2 - Xw1)*(Yw - Yf) + 2*(Xf2 - Xw2)*Yf
        
        A_near = 2*(Xw2 - Xw1)*Yw - (np.pi*D*D/4)
        
        N_base = A_base*rho_base
        N_far = A_far*rho_far
        N_wake = A_wake*rho_wake
        N_near = A_near*rho_near
        
        points_base = []
        while(len(points_base) < N_base):
            x_base = dom_coords[0:1, :] + (dom_coords[1:2, :] - dom_coords[0:1, :]) * np.random.rand(nsamp, 2)
            
            for i in range(x_base.shape[0]):
                if(len(points_base) < N_base):
                    if (x_base[i,0] <= Xb):
                        points_base.append(x_base[i,:])
                    elif ((x_base[i,0] > Xb) and (x_base[i,1]**2 >= Yb**2)):  
                else:
                    break
        
        points_base = np.asarray(points_base)
        
        points_far = []
        while(len(points_far) < N_far):
            x_far = dom_coords[0:1, :] + (dom_coords[1:2, :] - dom_coords[0:1, :]) * np.random.rand(nsamp, 2)
            
            for i in range(x_far.shape[0]):
                if(len(points_far) < N_far):
                    if ((x_far[i,0] <= Xf1) and (x_far[i,0] > Xb) and (x_far[i,1]*2 <= Yb**2)) :
                        points_far.append(x_far[i,:])
                    elif ((x_far[i,0] <= Xf2) and (x_far[i,0] > Xf1) and (x_far[i,1]*2 <= Yb**2) and (x_far[i,1]*2 > Yf**2)):
                        points_far.append(x_far[i,:])
                    elif ((x_far[i,0] > Xf2) and (x_far[i,1]*2 <= Yb**2)):
                        points_far.append(x_far[i,:])
                else:
                    break
        
        points_far = np.asarray(points_far)
        
        
        
        
        X_anchor = ''







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