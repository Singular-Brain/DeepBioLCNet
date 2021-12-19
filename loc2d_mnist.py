
import torch
from torch.nn.modules.utils import _pair


from tqdm import tqdm
import os
from bindsnet.network.monitors import Monitor
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from tqdm import tqdm

# from bindsnet.analysis.plotting import (
#     plot_input,
#     plot_spikes,
#     plot_voltages,
# )

from temp import plot_locally_connected_feature_maps

from time import time as t
from torchvision import transforms
from bindsnet.learning import PostPre

from bindsnet.network.nodes import AdaptiveLIFNodes
from bindsnet.network.nodes import Input
from bindsnet.network.network import Network
from bindsnet.network.topology import Connection, LocalConnection2D
from bindsnet.encoding import PoissonEncoder
from bindsnet.datasets import MNIST


# Hyperparameters
in_channels = 2
n_filters = 50
input_shape = [20, 20]
kernel_size = _pair(12)
stride = _pair(4)
tc_theta_decay = 1e7
theta_plus = 0.05
norm = 0.2*kernel_size[0]*kernel_size[1]
wmin = 0.0
wmax = 1.0
nu = (1e-4, 1e-2)
inh = 25.0
dt = 1.0
time = 250
intensity = 128
n_epochs = 1
n_train = 500
progress_interval = 10
batch_size = 10

plot = True
slice_to_plot = 0


# Build network
network = Network()

input_layer = Input(
    shape=[in_channels, input_shape[0], input_shape[1]],
    traces=True,
    tc_trace=20
)

compute_conv_size = lambda inp_size, k, s: int((inp_size - k) / s) + 1
conv_size = _pair(compute_conv_size(input_shape[0], kernel_size[0], stride[0]))

output_layer = AdaptiveLIFNodes(
    shape=[n_filters, conv_size[0], conv_size[1]],
    traces=True,
    rest=-65.0,
    reset=-60.0,
    thresh=-52.0,
    refrac=5,
    tc_decay=100.0,
    tc_trace=20.0,
    theta_plus=theta_plus,
    tc_theta_decay=tc_theta_decay,
)

input_output_conn = LocalConnection2D(
    input_layer,
    output_layer,
    kernel_size=kernel_size,
    stride=stride,
    in_channels=in_channels,
    out_channels=n_filters,
    input_shape=[20,20],
    nu=nu,
    update_rule=PostPre,
    wmin=wmin,
    wmax=wmax,
    norm=norm,
)

w_inh_LC = torch.zeros(n_filters, conv_size[0], conv_size[0], n_filters, conv_size[0], conv_size[0])
for c in range(n_filters):
    for w1 in range(conv_size[0]):
        for w2 in range(conv_size[0]):
            w_inh_LC[c, w1, w2, :, w1, w2] = -inh
            w_inh_LC[c, w1, w2, c, w1, w2] = 0

w_inh_LC = w_inh_LC.reshape(output_layer.n, output_layer.n)
recurrent_conn = Connection(output_layer, output_layer, w=w_inh_LC)

network.add_layer(input_layer, name="X")
network.add_layer(output_layer, name="Y")
network.add_connection(input_output_conn, source="X", target="Y")
network.add_connection(recurrent_conn, source="Y", target="Y")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = True
seed = 0
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

if gpu:
    network.to("cuda")

# Load MNIST data.
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../data/MNIST",
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.CenterCrop((input_shape[0], input_shape[1])), transforms.Lambda(lambda x: x * intensity)]
    ),
)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Train the network.
print("Begin training.\n")
start = t()

inpt_axes = None
inpt_ims = None
spike_ims = None
spike_axes = None
weights1_im = None
voltage_ims = None
voltage_axes = None



for epoch in range(n_epochs):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(train_dataloader)):
        # Get next input sample.
        if step > n_train:
            break
        inputs = {"X": batch["encoded_image"].view(time, batch_size, 1, input_shape[0], input_shape[1]).repeat(1,1,2,1,1)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        label = batch["label"]

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Optionally plot various simulation information.
        if plot:
            # image = batch["image"].view(input_shape[0], input_shape[1])

            # inpt = inputs["X"].view(time, input_shape[0]*input_shape[1]).sum(0).view(input_shape[0], input_shape[1])
            weights1 = input_output_conn.w
            #print(weights1.shape)
            # _spikes = {
            #     "X": spikes["X"].get("s").view(time, -1),
            #     "Y": spikes["Y"].get("s").view(time, -1),
            # }
            # _voltages = {"Y": voltages["Y"].get("v").view(time, -1)}

            # inpt_axes, inpt_ims = plot_input(
            #     image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
            # )
            # spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights1_im = plot_locally_connected_feature_maps(weights1, n_filters, in_channels, slice_to_plot, input_shape[0], kernel_size[0], conv_size[0], im=weights1_im)
            # voltage_ims, voltage_axes = plot_voltages(
            #     _voltages, ims=voltage_ims, axes=voltage_axes
            # )

            plt.pause(1)

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Training complete.\n")

weights1 = input_output_conn.w
weights1_im = plot_locally_connected_feature_maps(weights1, n_filters, in_channels, slice_to_plot, input_shape[0], kernel_size[0], conv_size[0])
plt.savefig('test.png')
plt.pause(100)