# -*- coding: utf-8 -*-

from bindsnet.network.nodes import Nodes
import os
import torch
import random
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import collections
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import torch.nn.functional as fn

from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Sequence
from torch.nn.modules.utils import _pair

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network, load
from bindsnet.network.nodes import Input, LIFNodes, AdaptiveLIFNodes, IFNodes
from bindsnet.network.topology import LocalConnection, Connection, LocalConnectionOrig, MaxPool2dLocalConnection
from bindsnet.network.monitors import Monitor, AbstractMonitor
from bindsnet.learning import PostPre, MSTDP, MSTDPET, WeightDependentPostPre, Hebbian
from bindsnet.learning.reward import DynamicDopamineInjection, DopaminergicRPE
from bindsnet.analysis.plotting import plot_locally_connected_weights,plot_locally_connected_weights_meh,plot_spikes,plot_locally_connected_weights_meh2,plot_convergence_and_histogram,plot_locally_connected_weights_meh3
from bindsnet.analysis.visualization import plot_weights_movie, plot_spike_trains_for_example,summary, plot_voltage
from bindsnet.utils import reshape_locally_connected_weights, reshape_locally_connected_weights_meh, reshape_conv2d_weights


import json
with open('./config.json') as f:
    config = json.load(f)


if torch.cuda.is_available():
    device =  torch.device("cuda")
    gpu = True
else:
    device =  torch.device("cpu")
    gpu = False

def manual_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


SEED = 2045 # The Singularity is Near!
manual_seed(SEED)
WANDB = config["WANDB"]

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

if WANDB:
    import wandb


"""## Kernel """

class AbstractKernel(ABC):
    def __init__(self, kernel_size):
        """
        Base class for generating image filter kernels such as Gabor, DoG, etc. Each subclass should override :attr:`__call__` function.
        Instantiates a ``Filter Kernel`` object.
        :param window_size : The size of the kernel (int)
        """
        self.window_size = kernel_size

    def __call__(self):
        pass

class DoGKernel(AbstractKernel):
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], sigma1 : float, sigma2 : float):
        """
        Generates DoG filter kernels.
        :param kernel_size: Horizontal and vertical size of DOG kernels.(If pass int, we consider it as a square filter) 
        :param sigma1 : The sigma parameter for the first Gaussian function.
        :param sigma2 : The sigma parameter for the second Gaussian function.
        """
        super(DoGKernel, self).__init__(kernel_size)
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
    def __call__(self):
        k = self.window_size//2
        x, y = np.mgrid[-k:k+1:1, -k:k+1:1]
        a = 1.0 / (2 * math.pi)
        prod = x*x + y*y
        f1 = (1/(self.sigma1*self.sigma1)) * np.exp(-0.5 * (1/(self.sigma1*self.sigma1)) * (prod))
        f2 = (1/(self.sigma2*self.sigma2)) * np.exp(-0.5 * (1/(self.sigma2*self.sigma2)) * (prod))
        dog = a * (f1-f2)
        dog_mean = np.mean(dog)
        dog = dog - dog_mean
        dog_max = np.max(dog)
        dog = dog / dog_max
        dog_tensor = torch.from_numpy(dog)
        # returns a 2d tensor corresponding to the requested DoG filter
        return dog_tensor.float()

class Filter():
    """
    Applies a filter transform. Each filter contains a sequence of :attr:`FilterKernel` objects.
    The result of each filter kernel will be passed through a given threshold (if not :attr:`None`).
    Args:
        filter_kernels (sequence of FilterKernels): The sequence of filter kernels.
        padding (int, optional): The size of the padding for the convolution of filter kernels. Default: 0
        thresholds (sequence of floats, optional): The threshold for each filter kernel. Default: None
        use_abs (boolean, optional): To compute the absolute value of the outputs or not. Default: False
    .. note::
        The size of the compund filter kernel tensor (stack of individual filter kernels) will be equal to the 
        greatest window size among kernels. All other smaller kernels will be zero-padded with an appropriate 
        amount.
    """
    # filter_kernels must be a list of filter kernels
    # thresholds must be a list of thresholds for each kernel
    def __init__(self, filter_kernels, padding=0, thresholds=None, use_abs=False):
        tensor_list = []
        self.max_window_size = 0
        for kernel in filter_kernels:
            if isinstance(kernel, torch.Tensor):
                tensor_list.append(kernel)
                self.max_window_size = max(self.max_window_size, kernel.size(-1))
            else:
                tensor_list.append(kernel().unsqueeze(0))
                self.max_window_size = max(self.max_window_size, kernel.window_size)
        for i in range(len(tensor_list)):
            p = (self.max_window_size - filter_kernels[i].window_size)//2
            tensor_list[i] = fn.pad(tensor_list[i], (p,p,p,p))

        self.kernels = torch.stack(tensor_list)
        self.number_of_kernels = len(filter_kernels)
        self.padding = padding
        if isinstance(thresholds, list):
            self.thresholds = thresholds.clone().detach()
            self.thresholds.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
        else:
            self.thresholds = thresholds
        self.use_abs = use_abs

    # returns a 4d tensor containing the flitered versions of the input image
    # input is a 4d tensor. dim: (minibatch=1, filter_kernels, height, width)
    def __call__(self, input):

        # if input.dim() == 3:
        #     input2 = torch.unsqueeze(input, 0)
        input.unsqueeze_(0)
        output = fn.conv2d(input, self.kernels, padding = self.padding).float()
        if not(self.thresholds is None):
            output = torch.where(output < self.thresholds, torch.tensor(0.0, device=output.device), output)
        if self.use_abs:
            torch.abs_(output)
        return output.squeeze(0)

"""# Design network"""

compute_size = lambda inp_size, k, s: int((inp_size-k)/s) + 1
def convergence(c):
    if c.norm is None:
        return 1-torch.mean((c.w-c.wmin)*(c.wmax-c.w))/((c.wmax-c.wmin)/2)**2
    else:
        mean_norm_factor = c.norm / c.w.shape[-1]
        return  1-(torch.mean((c.w-c.wmin)*(c.wmax-c.w))/((c.wmax-c.wmin)/2)**2)


class LCNet(Network):
    def __init__(
        self,
        dt: float,
        time: int,
        channels: list,
        filters: list,
        strides: list,
        input_channels: int,
        nu_LC: Union[float, Tuple[float, float]],
        NodesType,
        crop_size:int,
        inh_type: bool,
        nu_inh: float,
        inh_factor: float,
        norm_factor: float,
        update_rule,
        update_rule_inh,
        wmin: float,
        wmax: float ,
        soft_bound: bool,
        theta_plus: float,
        tc_theta_decay: float,
        tc_trace:int,
        trace_additive,
        wandb_active:bool,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for class ``BioLCNet``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``(adaptive)LIFNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``(adaptive)LIFNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        manual_seed(SEED)
        super().__init__(dt=dt, reward_fn = None, online=True)
        

        self.dt = dt
        self.time = time
        self.input_channels = input_channels
        self.channels = channels
        self.filters = filters
        self.strides = strides
        self.update_rule = update_rule
        self.nu_LC = nu_LC
        self.NodesType = NodesType
        self.crop_size = crop_size
        self.inh_type = inh_type
        self.inh_factor = inh_factor
        self.clamp_intensity = kwargs.get('clamp_intensity',None)
        self.soft_bound = soft_bound
        self.norm_factor =norm_factor
        self.convergences = {}
        self.wmin = wmin 
        self.wmax = wmax
        self.wandb_active = wandb_active
        self.epochs_trained = 0

        ### nodes
        inp = Input(shape= [input_channels,crop_size,crop_size], traces=True, tc_trace=tc_trace,traces_additive = trace_additive)
        self.add_layer(inp, name="layer0")

        last_layer_size = crop_size
        last_layer = inp
        last_layer_channels = 1

        assert type(self.channels) == list
        network_hparams = [self.channels, self.filters, self.strides,
                           self.norm_factor, self.wmin, self.wmax,
                           self.soft_bound, self.update_rule, self.nu_LC,
                           self.NodesType, self.inh_type, self.inh_factor]
        network_hparams_list = []
        for hparam in network_hparams:
            if type(hparam) == list:
                network_hparams_list.append(hparam)
            else:
                network_hparams_list.append([hparam] * len(channels))
        assert all([len(hparam)==len(channels) for hparam in network_hparams_list])
        for i, (
                    n_channels, filter_size, stride, norm_LC,
                    wmin, wmax, soft_bound, update_rule, nu_LC,
                    NodesType, inh_type, inh_factor
                ) in enumerate(zip(*network_hparams_list)
                ):
            new_layer_size = compute_size(last_layer_size, filter_size, stride)
            new_layer = NodesType(
                shape= [n_channels, new_layer_size, new_layer_size],
                traces=True, tc_trace=tc_trace,traces_additive = trace_additive,
                tc_theta_decay = tc_theta_decay, theta_plus = theta_plus
                )
            self.add_layer(new_layer, name=f"layer{i+1}")
            ### connections 
            LC_connection = LocalConnection(
                last_layer, new_layer,
                filter_size, stride,
                last_layer_channels, n_channels,
                input_shape = [last_layer_size]*2,
                nu = nu_LC, update_rule= update_rule,
                wmin = wmin, wmax= wmax, soft_bound = soft_bound,
                norm = norm_LC)
            self.add_connection(LC_connection, f"layer{i}", f"layer{i+1}")
            ###
            last_layer_size = new_layer_size
            last_layer = new_layer
            last_layer_channels = n_channels
            ### Inhibitory connection
            if inh_type == 'recurrent':
                w_inh_LC = torch.zeros(n_channels,new_layer_size,new_layer_size,n_channels,new_layer_size,new_layer_size)
                for c in range(n_channels):
                    for w1 in range(new_layer_size):
                        for w2 in range(new_layer_size):
                            w_inh_LC[c,w1,w2,:,w1,w2] = - inh_factor
                            w_inh_LC[c,w1,w2,c,w1,w2] = 0
            
                w_inh_LC = w_inh_LC.reshape(new_layer.n,new_layer.n)
                                                            
                LC_recurrent_inhibition = Connection(
                    source=new_layer,
                    target=new_layer,
                    w=w_inh_LC,
                    nu= nu_inh,
                )
                self.add_connection(LC_recurrent_inhibition, f"layer{i+1}", f"layer{i+1}")

            elif inh_type == 'lateral':
                inh_neurons_layer = LIFNodes(
                    shape= [n_channels],
                    traces=True, tc_trace=tc_trace,traces_additive = trace_additive,
                    tc_theta_decay = tc_theta_decay, theta_plus = theta_plus
                    ).to(device)
                self.add_layer(inh_neurons_layer, name=f"layer{i+1}_inh_neurons")

                w_inh_LC = torch.zeros(n_channels, n_channels,new_layer_size,new_layer_size)
                w_inh_LC_rev = torch.zeros(n_channels, new_layer_size, new_layer_size, n_channels)

                for c in range(n_channels):
                    for w1 in range(new_layer_size):
                        for w2 in range(new_layer_size): 
                            w_inh_LC[c,c,w1,w2] = -inh_factor
                            w_inh_LC_rev[c,w1,w2,c] = 1/(new_layer_size**2)

                w_inh_LC = w_inh_LC.reshape(n_channels, new_layer.n)
                w_inh_LC_rev = w_inh_LC_rev.reshape(new_layer.n, n_channels)

                LC_lateral_inhibition = Connection(
                    source=inh_neurons_layer,
                    target=new_layer,
                    w=w_inh_LC,
                    nu= nu_inh,
                ).to(device)

                LC_lateral_inhibition_rev = Connection(
                    source=new_layer,
                    target=inh_neurons_layer,
                    w=w_inh_LC_rev,
                    nu= nu_inh,
                ).to(device)

                self.add_connection(LC_lateral_inhibition, f"layer{i+1}_inh_neuron", f"layer{i+1}")
                self.add_connection(LC_lateral_inhibition_rev, f"layer{i+1}",f"layer{i+1}_inh_neuron")

        self.to(device)


    def fit(
        self,
        dataloader,
        n_train = 2000,
        verbose = True,
    ):
        manual_seed(SEED)
        self.verbose = verbose
        if self.wandb_active:
            wandb.watch(self)
            
        # add Monitors
        #Plot_et = PlotET(i = 0, j = 0, source = self.layers["main"], target = self.layers["output"], connection = self.connections[("main","output")])
        #self.add_monitor(Plot_et, name="Plot_et")

        self.spikes = {}
        for layer in set(self.layers):
            self.spikes[layer] = Monitor(self.layers[layer], state_vars=["s"], time=None)
            self.add_monitor(self.spikes[layer], name="%s_spikes" % layer)

        pbar = tqdm(total=n_train)
        self.reset_state_variables()

        for (i, datum) in enumerate(dataloader):
            if i>=n_train:
                break

            image = datum["encoded_image"]
            # Run the network on the input.
            if gpu:
                inputs = {"layer0": image.cuda().view(self.time, 1, self.input_channels, self.crop_size, self.crop_size)}
            else:
                inputs = {"layer0": image.view(self.time, 1, self.input_channels, self.crop_size, self.crop_size)}
            ### Spike clamping (baseline activity)
            clamp = {}
            if self.clamp_intensity is not None:
                encoder = PoissonEncoder(time = self.time, dt = self.dt)
                clamp['output'] = encoder.enc(datum = torch.rand(self.layers['output'].n)*self.clamp_intensity,time = self.time, dt = self.dt)

            self.run(inputs=inputs, 
                    time=self.time, 
                    one_step=True,
                    clamp = clamp
                     )

            # Get voltage recording.

            if verbose:
                print(f"\r ",
                    f"input_mean_fire_freq: {torch.mean(image.float())*1000:.1f}",
                    '- Spikes:', {name: monitor.get('s').sum().item() for name, monitor in self.spikes.items()},
                    '- Layers Mean:', list(map(lambda x: {x[0]: round(x[1].w.mean().item(),3)}, self.connections.items())),
                    '- convergences:', {name: round(convergence(c).item() , 5) for name, c in self.connections.items() if c.wmin!=-np.inf and c.wmax!=np.inf },
                    '- std:', {name: round(c.w.std().item() , 5) for name, c in self.connections.items() if c.wmin!=-np.inf and c.wmax!=np.inf },
                    end = ''
                    )
                   
            if self.wandb_active:
                wandb.log({
                        **{' to '.join(name) + ' convergence': convergence(c).item() for name, c in self.connections.items() if name[0]!=name[1]},
                        **{' to '.join(name) + ' std': c.w.std().item() for name, c in self.connections.items() if name[0]!=name[1]},
                        **{name + ' spikes': monitor.get('s').sum().item() for name, monitor in self.spikes.items()},
                        **{' to '.join(name) + " gradients": wandb.Histogram(c.w.cpu()) for name, c in self.connections.items() if name[0]!=name[1]},
                    },
                    step = self.epochs_trained)
            #Plot_et.plot()    
            self.reset_state_variables()  # Reset state variables.
            
            # pbar.set_description_str("")
            pbar.update()
            self.epochs_trained +=1


    def predict(
        self,
        val_loader,
        n_pred,
    ):
        manual_seed(SEED)

        monitors = []
        for layer in self.layers:
            if layer == 'layer0':
                continue
            monitor = Monitor(self.layers[layer], ["s", "v"])
            monitors.append(monitor)
            self.add_monitor(monitor, name=f"{layer} monitor")
        self.train(False)
        S= []
        V= []
        y= []
        pbar = tqdm(total=n_pred)
        for (i, datum) in enumerate(val_loader):
            if i > n_pred:
                break

            image = datum["encoded_image"]
            if self.label is None : 
              label = datum["label"]
            else :
              label = self.label

            # Run the network on the input.
            if gpu:
                inputs = {"layer0": image.cuda().view(self.time, 1, self.input_channels, self.crop_size, self.crop_size)}
            else:
                inputs = {"layer0": image.view(self.time, 1, self.input_channels, self.crop_size, self.crop_size)}

            self.run(inputs=inputs, 
                    time=self.time, 
                    one_step = True,
                    true_label = label.int().item(),
                    dopaminergic_layers= self.dopaminergic_layers,
                     )

            S.append(list(map(lambda x: x.get('s'), monitors)))
            V.append(list(map(lambda x: x.get('v'), monitors)))
            y.append(label)

            self.reset_state_variables()  # Reset state variables.
            pbar.update()

        self.train(True)
        return S, V ,y

"""# Load Dataset"""

class ClassSelector(torch.utils.data.sampler.Sampler):
    """Select target classes from the dataset"""
    def __init__(self, target_classes, data_source, mask = None):
        if mask is not None:
            self.mask = mask
        else:
            self.mask = torch.tensor([1 if data_source[i]['label'] in target_classes else 0 for i in range(len(data_source))])
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)

kernels = [DoGKernel(7,1,2),
			DoGKernel(7,2,1),]
filter = Filter(kernels, padding = 3, thresholds = 50/255)
# Load MNIST data.

def load_datasets(network_hparams, data_hparams, mask=None, test_mask=None):
    manual_seed(SEED)
    dataset = MNIST(
        PoissonEncoder(time=network_hparams['time'], dt=network_hparams['dt']),
        None,
        root=os.path.join("..", "..", "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [
                transforms.CenterCrop(data_hparams['crop_size']),
                transforms.ToTensor(),
                # filter,
                transforms.Lambda(lambda x: (
                    x.round() if data_hparams['round_input'] else x
                ) * data_hparams['intensity']),
            ]
        ),
    )

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                            sampler = ClassSelector(
                                                    target_classes = target_classes,
                                                    data_source = dataset,
                                                    mask = mask,
                                                    ) if target_classes else None
                                            )

    # Load test dataset
    test_dataset = MNIST(   
        PoissonEncoder(time=network_hparams['time'], dt=network_hparams['dt']),
        None,
        root=os.path.join("..", "..", "data", "MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (
                    x.round() if data_hparams['round_input'] else x
                ) * data_hparams['intensity']),
                transforms.CenterCrop(data_hparams['crop_size'])
                # filter,
                ]
        ),
    )

    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                            sampler = ClassSelector(
                                                    target_classes = target_classes,
                                                    data_source = test_dataset,
                                                    mask = mask_test,
                                                    ) if target_classes else None
                                            )
    

    return dataloader, val_loader

"""# Set up hyper-parameters"""

# Dataset Hyperparameters
target_classes = None #(0,1)
if target_classes:
    npz_file = np.load(f'bindsnet/mask_{"_".join([str(i) for i in target_classes])}.npz')
    # npz_file = np.load('bindsnet/mask_0_1.npz') ##### KESAFAT KARI !!!
    mask, mask_test = torch.from_numpy(npz_file['arr_0']), torch.from_numpy(npz_file['arr_1'])
    n_classes = len(target_classes)
    
else:
    mask = None
    mask_test = None
    n_classes = 10

data_hparams = { 
    'intensity': config['intensity'],
    'crop_size': config['crop_size'],
    'round_input': config['round_input'],
}

network_hparams = {
    # net structure 
    'channels': config['channels'],
    'filters': config['filters'],
    'strides': config['strides'],
    'input_channels': config['input_channels'],
    # time & Phase
   'time': config['time'],
    'dt' : config['dt'],
    # Nodes
    'NodesType': eval(config['NodesType']),
    'theta_plus': config['theta_plus'],
    'tc_theta_decay': config['tc_theta_decay'],
    'tc_trace':config['tc_trace'],
    'trace_additive' : config['trace_additive'],
    # Learning
    'update_rule': eval(config['update_rule']),
    'update_rule_inh' : config['update_rule_inh'],
    'nu_LC': config['nu_LC'],
    'soft_bound': config['soft_bound'],
    'wmin': config['wmin'],
    'wmax': config['wmax'],
    # Inhibition
    'nu_inh': config['nu_inh'],
    'inh_LC': config['inh_LC'],
    'inh_type': config['inh_type'],
    'inh_factor': config['inh_factor'],
    # Normalization
    'norm_factor': config['norm_factor'],
    # clamp
    'clamp_intensity': config['clamp_intensity'],
}


dataloader, val_loader = load_datasets(network_hparams, data_hparams, mask, mask_test)

"""# Training"""

manual_seed(SEED)
if WANDB:
    wandb.init(project='SemanticPooling', entity='singularbrain', config=network_hparams)
net = LCNet(**network_hparams, **data_hparams, wandb_active = WANDB)
net

net.fit(n_train = 2_000, dataloader = dataloader)

"""**Save Model:**"""
net.save(config.save_name)

