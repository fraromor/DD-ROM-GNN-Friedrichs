import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

## PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

## Pytorch geometric
import torch_geometric
import torch_geometric.nn as nng
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.loader import ClusterData, ClusterLoader

# Torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

CHECKPOINT_PATH = "./tb_logs/"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = False
torch.backends.cudnn.benchmark = False

n_batch = 100
NUS = 2

# domain decomposition in 4 subdomains
procs = [50687, 82943, 126719, 175103]
up = [procs[0] + 1, procs[1] + 1, procs[2] + 1, procs[3] + 1]
down = [0, procs[0]+1, procs[1]+1, procs[2]+1]

class VVGRAPH(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        n_inputs = 15
        nn1 = nn.Sequential(nn.Linear(2, 12), nn.ReLU(),
                        nn.Linear(12, n_inputs * 18)).double()
        nn2 = nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
                            nn.Linear(8, 30 * 1)).double()

        self.conv0 = nng.Sequential('x, edge_index, edge_attr', [
            # (nng.ChebConv(3, 8, 10), 'x, edge_index -> x'),
            (nng.NNConv(n_inputs, 18, nn1, aggr='mean').double(), 'x, edge_index, edge_attr -> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(18, 21, aggr='mean', project=True).double(), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(21, 24, aggr='mean', project=True).double(), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(24, 27, aggr='mean', project=True).double(), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.SAGEConv(27, 30, aggr='mean', project=True).double(), 'x, edge_index-> x'),
            nn.ReLU(inplace=True),
            (nng.NNConv(30, 1, nn2, aggr='mean').double(), 'x, edge_index, edge_attr -> x'),
        ])
                    
        # Saving hyperearameters of autoencoder
        self.save_hyperparameters()
        
        # inputs used to derive the architecture summary
        channels = 15
        x = torch.tensor(np.arange(channels * 2),
                         dtype=torch.float64).reshape(2, channels)
        
        e = torch.tensor(np.arange(2),
                         dtype=torch.long).reshape(2, 1)
        e_attr = e.T.to(dtype=torch.float64)
        self.example_input_array = [x, e, e_attr, torch.ones(2)]

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
    
    def _get_reconstruction_loss(self, batch, batch_idx):
        x, x_edge_index, x_edge_attr, y = batch.x, batch.edge_index, batch.edge_attr, batch.y
        
        x_hat = self.forward(x, x_edge_index, x_edge_attr, batch.avg).reshape(-1)
        diff = (x_hat.reshape(-1, 1)-y).reshape(-1)
        
        loss = torch.linalg.norm(diff, dim=0)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=5e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_edge_index, x_edge_attr, y = batch.x, batch.edge_index, batch.edge_attr, batch.y
        x_hat = self.forward(x, x_edge_index, x_edge_attr, batch.avg)
        loss = torch.linalg.norm(y-x_hat, dim=0)/torch.linalg.norm(y, dim=0)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)

    def on_train_epoch_end(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params,
                                                 self.current_epoch)


def train_vv(train_loader, val_loader, logger):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, f"vv"),
        max_epochs=50,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=1),
        ],
        logger=logger,
        num_sanity_val_steps=0,
        devices=[1])

    trainer.logger._log_graph = (
        None  # If True, we plot the computation graph in tensorboard
    )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"vv.ckpt")
    # pretrained_filename = "./arch/vv_no_mass_aug_adv_all_fid.ckpt"
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = VVGRAPH.load_from_checkpoint(pretrained_filename)
        trainer.fit(model, train_loader, val_loader)
    else:
        model = VVGRAPH()
        print("Training...")
        trainer.fit(model, train_loader, val_loader)

    return model

def feature_augmentation(x, laplace, adv, op_adv0, op_adv1):
    l = x[:, 0].reshape(-1, 1)
    m = x[:, 1].reshape(-1, 1)
    inp = torch.cat((x, m-l), dim=1)

    outputs_list = list()
    outputs_list.append(inp)
    outputs_list.append(torch.sparse.mm(laplace, inp))
    outputs_list.append(torch.sparse.mm(adv,inp))
    outputs_list.append(torch.sparse.mm(op_adv0, inp))
    outputs_list.append(torch.sparse.mm(op_adv1, inp))

    ret = torch.cat(outputs_list, dim=1)
    return ret

def prepare_data_single_subdomain(): 
    # Load inputs outputs
    true_inputs = torch.load("./data/inp_all_together.pt")
    true_output = torch.load("./data/out_all_together.pt")
    low_fidelity_snaps = true_inputs[:, :2, :]
    output = true_output[:, 0, :]
    print("Loaded true input, output ", true_inputs.shape, true_output.shape)

    n_nodes_inputs = low_fidelity_snaps.shape[2]
    n_nodes_output = output.shape[1]
    n_train = low_fidelity_snaps.shape[0]
    channels = low_fidelity_snaps.shape[1]
    print("Loaded shape: ", low_fidelity_snaps.shape, output.shape, channels)
    print("n nodes: ", n_nodes_inputs, n_nodes_output, "n train: ", n_train)
    
    # Load operators
    mass_ = torch.load("./operators/mass.pt")
    laplace_ = torch.load("./operators/laplace.pt")
    adv0_ = torch.load("./operators/adv0.pt")
    adv1_ = torch.load("./operators/adv1.pt")
    adv_ = torch.load("./operators/adv.pt")

    mass = torch.sparse_coo_tensor(mass_[0], mass_[1],
                                        (true_output.shape[2], true_output.shape[2]))
    laplace = torch.sparse_coo_tensor(laplace_[0], laplace_[1],
                                        (true_output.shape[2], true_output.shape[2]))
    adv = torch.sparse_coo_tensor(adv_[0], adv_[1],
                                        (true_output.shape[2], true_output.shape[2]))
    adv0 = torch.sparse_coo_tensor(adv0_[0], adv0_[1],
                                        (true_output.shape[2], true_output.shape[2]))
    adv1 = torch.sparse_coo_tensor(adv1_[0], adv1_[1],
                                        (true_output.shape[2], true_output.shape[2]))
    
    # average of training snapshots only, the same as the variable 'train_idxs'
    avg = torch.mean(output[::5], dim=0) 
    print("Avg: ", avg.shape)
    torch.save(avg, "avg.pt")

    edge_index = mass_[0]
    print("edge: ", edge_index.shape)
            
    edge_attr = torch.load("./operators/edge_attr_finer.pt")
    print("Edge attributes: ", edge_attr.shape)
    
    data = [
        Data(x=feature_augmentation(
                low_fidelity_snaps[i].reshape(channels, -1).T, #0-2
                laplace, #3-5
                adv, #6-8
                adv0, #9-11
                adv1), #12-14
        edge_index=edge_index.contiguous(),
        edge_attr=edge_attr,
        y=output[i].reshape(-1, 1),
        avg=avg.detach())
        for i in range(low_fidelity_snaps.shape[0])
    ]

    train_idxs = torch.Tensor([x for x in range(100) if x%5==0]).to(dtype=torch.int64)
    print("train and val indices: ", train_idxs.shape)
    
    batch = Batch.from_data_list([data[i] for i in train_idxs])
    print("Batch: ", batch)
    
    train_data = ClusterData(batch, num_parts=100000, save_dir="./cluster/", recursive=True)
    print("ClusterData: ", train_data.data)
    
    train_loader = ClusterLoader(train_data, batch_size=n_batch, shuffle=True)
    print("ClusterLoader: ", train_loader)

    # set the validation set only for one of the 4 subdomains
    n_subdomain = 0
    edge_index_val = torch.load("./operators/sparsity_pattern_finer_" + str(n_subdomain) + ".pt")
    edge_attr_finer = torch.load("./operators/edge_attr_finer_"+str(n_subdomain)+".pt")
    aug_val = torch.load("./data/inp_augmented_"+str(n_subdomain)+".pt")
    print("Validation data augmented: ", aug_val[0].shape, laplace.shape, channels)
    
    val = [
        Data(x=aug_val[i].reshape(15, -1).T,#feature_augmentation(aug_val[i][:3],laplace,adv,adv0,adv1),
        edge_index=edge_index_val.contiguous(),
        edge_attr=edge_attr_finer,
        avg=avg[down[n_subdomain]:up[n_subdomain]],
        y=output[i].reshape(-1, 1)[down[n_subdomain]:up[n_subdomain]])
        for i in range(low_fidelity_snaps.shape[0])
    ]

    # set validation set 
    val_idx = torch.arange(3, 100, 20).to(dtype=torch.int64)
    val_loader = DataLoader([val[i] for i in val_idx], batch_size=1, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    print("Training with pytorch version: ", torch.__version__)
    print("Training with pytorch geometric version: ",
          torch_geometric.__version__)
    print("Training with pytorch lightning version: ", pl.__version__)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    
    # viscosity levels
    nus = [0.05, 0.01, 0.0005]
    
    # prepare data to pass to trainer
    train_loader, val_loader = prepare_data_single_subdomain()
    logger = TensorBoardLogger("tb_logs", name="vv")
    
    # train the GNN
    model = train_vv(train_loader, val_loader, logger)
    
