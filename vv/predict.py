import time
import warnings
warnings.filterwarnings("ignore")

## Imports for plotting
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

## PyTorch
import torch
import torch.nn as nn

## Pytorch geometric
import torch_geometric
import torch_geometric.nn as nng

# Torchvision
import pytorch_lightning as pl

CHECKPOINT_PATH = "./tb_logs/"

# Setting the seed
pl.seed_everything(42)
torch.set_default_tensor_type(torch.DoubleTensor)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = False
torch.backends.cudnn.benchmark = False

n_batch = 100
NUS = 2
n_nodes = 0
procs = [50687, 82943, 126719, 175103]
up = [procs[0] + 1, procs[1] + 1, procs[2] + 1, procs[3] + 1]
down = [0, procs[0]+1, procs[1]+1, procs[2]+1]

class VVGRAPHmodel(nn.Module):
    def __init__(self,
                 edge_index=None,
                 ops=None):
        super().__init__()
        
        n_inputs = 15
        nn1 = nn.Sequential(nn.Linear(2, 12), nn.ReLU(),
                        nn.Linear(12, n_inputs * 18))
        nn2 = nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
                            nn.Linear(8, 30 * 1))

        self.conv0 = nng.Sequential('x, edge_index, edge_attr', [
            (nng.NNConv(n_inputs, 18, nn1, aggr='mean').jittable(), 'x, edge_index, edge_attr -> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(18, 21, aggr='mean', project=True).jittable(), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(21, 24, aggr='mean', project=True).jittable(), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(24, 27, aggr='mean', project=True).jittable(), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(27, 30, aggr='mean', project=True).jittable(), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.NNConv(30, 1, nn2, aggr='mean').jittable(), 'x, edge_index, edge_attr -> x'),
        ])

    def forward(self, x, x_edge_index, x_edge_attr, avg_g):
        # augment inputs with all operators except mass        
        out = nn.functional.relu(
            # highest input fidelity
            x[:, 1].reshape(-1, 1) +\
            # correction
            self.conv0(x, x_edge_index, x_edge_attr)+\
            # average training snapshot
            avg_g.reshape(-1, 1))
        return out

class VVGRAPH(pl.LightningModule):
    def __init__(self,
                 edge_index=None,
                 ops=None):
        super().__init__()
        
        n_inputs = 15
        nn1 = nn.Sequential(nn.Linear(2, 12), nn.ReLU(),
                        nn.Linear(12, n_inputs * 18))
        nn2 = nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
                            nn.Linear(8, 30 * 1))

        self.conv0 = nng.Sequential('x, edge_index, edge_attr', [
            # (nng.ChebConv(3, 8, 10), 'x, edge_index -> x'),
            (nng.NNConv(n_inputs, 18, nn1, aggr='mean'), 'x, edge_index, edge_attr -> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(18, 21, aggr='mean', project=True), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(21, 24, aggr='mean', project=True), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(24, 27, aggr='mean', project=True), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(27, 30, aggr='mean', project=True), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.NNConv(30, 1, nn2, aggr='mean'), 'x, edge_index, edge_attr -> x'),
        ])
                    
    def forward(self, x, x_edge_index, x_edge_attr, avg_g):
        out = nn.functional.relu(
            # highest input fidelity
            x[:, 1].reshape(-1, 1) +\
            # correction
            self.conv0(x, x_edge_index, x_edge_attr)+\
            # average training snapshot
            avg_g.reshape(-1, 1))
        return out
    
if __name__ == "__main__":
  print("Training with pytorch version: ", torch.__version__)
  print("Training with pytorch geometric version: ",
        torch_geometric.__version__)
  print("Training with pytorch lightning version: ", pl.__version__)

  device = torch.device(
      "cuda") if torch.cuda.is_available() else torch.device("cpu")
  print("Device:", device)

  nus = [0.05, 0.01, 0.0005]
  pretrained_filename = 'tb_logs/vv.ckpt'
  model = VVGRAPH.load_from_checkpoint(pretrained_filename)
  torch.save(model.state_dict(), "mod.pt")

  model = VVGRAPHmodel()
  model.load_state_dict(torch.load("mod.pt"), strict=False)
  model.eval()

  orig_output = torch.load("./data/out_all_together.pt")
  mass_ = torch.load("./operators/mass_norm.pt")
  mass = torch.sparse_coo_tensor(
      mass_[0], mass_[1],
      (orig_output.shape[2], orig_output.shape[2]))
  
  err = []
  print("Output: ", orig_output.shape)

  edge_index = mass_[0]
  edge_attr = torch.load("./operators/edge_attr_finer.pt")
  avg = torch.load("avg.pt")
  
#   jitmodel = torch.jit.script(model)
#   jitmodel.eval()

  print("start error evaluation")
  N = 100
  for i in range(N):
    inp = list()
    inp.append(torch.load("./inp_aug/inp"+str(i)+".pt"))
    inp = torch.cat(inp, dim=0)
    
    # predict. Can be also decomposed in subdomains
    s = time.time()
    forwarded = model.forward(inp,
                            edge_index,
                            edge_attr, avg).reshape(-1)
    print("Time online: ", time.time()-s)
    
    diff = (forwarded-orig_output[i, 0, :]).reshape(-1, 1)
    
    # evalute error with mass matrix
    with torch.no_grad():
      
      # L2 discrete norm
      e = torch.norm(diff)/torch.norm(orig_output[i, 0, :])
      
      # L2 norm with mass matrix of FEM spaces
      # e = torch.dot(
      #     torch.sparse.mm(mass, diff).reshape(-1), diff.reshape(-1))/\
      #     torch.dot(torch.sparse.mm(mass, orig_output[i, 0, :].reshape(-1, 1)).reshape(-1),
      #     orig_output[i, 0, :].reshape(-1))
      
    err.append(e.item())
    print("Relative prediction error for test with index {}: ".format(i), e)
    
    del diff
    del forwarded
    del e

ple = torch.Tensor(err)
torch.save(ple, "err.pt")

plt.semilogy(ple)
plt.grid(which='both')
plt.tight_layout()
plt.savefig("errors.pdf")
plt.close()
