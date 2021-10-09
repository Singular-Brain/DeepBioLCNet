import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

from abc import ABC
from typing import Union, Optional, Iterable, Dict

from .nodes import Nodes
from .topology import AbstractConnection

from bindsnet import manual_seed


class AbstractMonitor(ABC):
    # language=rst
    """
    Abstract base class for state variable monitors.
    """


class Monitor(AbstractMonitor):
    # language=rst
    """
    Records state variables of interest.
    """

    def __init__(
        self,
        obj: Union[Nodes, AbstractConnection],
        state_vars: Iterable[str],
        time: Optional[int] = None,
        batch_size: int = 1,
        device: str = "cpu",
    ):
        # language=rst
        """
        Constructs a ``Monitor`` object.

        :param obj: An object to record state variables from during network simulation.
        :param state_vars: Iterable of strings indicating names of state variables to record.
        :param time: If not ``None``, pre-allocate memory for state variable recording.
        :param device: Allow the monitor to be on different device separate from Network device
        """
        super().__init__()

        self.obj = obj
        self.state_vars = state_vars
        self.time = time
        self.batch_size = batch_size
        self.device = device

        # if time is not specified the monitor variable accumulate the logs
        if self.time is None:
            self.device = "cpu"

        self.recording = []
        self.reset_state_variables()

    def get(self, var: str) -> torch.Tensor:
        # language=rst
        """
        Return recording to user.

        :param var: State variable recording to return.
        :return: Tensor of shape ``[time, n_1, ..., n_k]``, where ``[n_1, ..., n_k]`` is the shape of the recorded state
        variable.
        Note, if time == `None`, get return the logs and empty the monitor variable

        """
        return_logs = torch.cat(self.recording[var], 0)
        # if self.time is None:
        #     self.recording[var] = []
        return return_logs

    def record(self, **kwargs) -> None:
        # language=rst
        """
        Appends the current value of the recorded state variables to the recording.
        """
        for v in self.state_vars:
            data = getattr(self.obj, v).unsqueeze(0)
            # self.recording[v].append(data.detach().clone().to(self.device))
            self.recording[v].append(
                torch.empty_like(data, device=self.device, requires_grad=False).copy_(
                    data, non_blocking=True
                )
            )
            # remove the oldest element (first in the list)
            if self.time is not None:
                self.recording[v].pop(0)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets recordings to empty ``List``s.
        """
        if self.time is None:
            self.recording = {v: [] for v in self.state_vars}
        else:
            self.recording = {
                v: [[] for i in range(self.time)] for v in self.state_vars
            }


class NetworkMonitor(AbstractMonitor):
    # language=rst
    """
    Record state variables of all layers and connections.
    """

    def __init__(
        self,
        network: "Network",
        layers: Optional[Iterable[str]] = None,
        connections: Optional[Iterable[str]] = None,
        state_vars: Optional[Iterable[str]] = None,
        time: Optional[int] = None,
    ):
        # language=rst
        """
        Constructs a ``NetworkMonitor`` object.

        :param network: Network to record state variables from.
        :param layers: Layers to record state variables from.
        :param connections: Connections to record state variables from.
        :param state_vars: List of strings indicating names of state variables to
            record.
        :param time: If not ``None``, pre-allocate memory for state variable recording.
        """
        super().__init__()

        self.network = network
        self.layers = layers if layers is not None else list(self.network.layers.keys())
        self.connections = (
            connections
            if connections is not None
            else list(self.network.connections.keys())
        )
        self.state_vars = state_vars if state_vars is not None else ("v", "s", "w")
        self.time = time

        if self.time is not None:
            self.i = 0

        # Initialize empty recording.
        self.recording = {k: {} for k in self.layers + self.connections}

        # If no simulation time is specified, specify 0-dimensional recordings.
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.Tensor()

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.Tensor()

        # If simulation time is specified, pre-allocate recordings in memory for speed.
        else:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.zeros(
                            self.time, *getattr(self.network.layers[l], v).size()
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.zeros(
                            self.time, *getattr(self.network.connections[c], v).size()
                        )

    def get(self) -> Dict[str, Dict[str, Union[Nodes, AbstractConnection]]]:
        # language=rst
        """
        Return entire recording to user.

        :return: Dictionary of dictionary of all layers' and connections' recorded
            state variables.
        """
        return self.recording

    def record(self, **kwargs) -> None:
        # language=rst
        """
        Appends the current value of the recorded state variables to the recording.
        """
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        data = getattr(self.network.layers[l], v).unsqueeze(0).float()
                        self.recording[l][v] = torch.cat(
                            (self.recording[l][v], data), 0
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        data = getattr(self.network.connections[c], v).unsqueeze(0)
                        self.recording[c][v] = torch.cat(
                            (self.recording[c][v], data), 0
                        )

        else:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        data = getattr(self.network.layers[l], v).float().unsqueeze(0)
                        self.recording[l][v] = torch.cat(
                            (self.recording[l][v][1:].type(data.type()), data), 0
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        data = getattr(self.network.connections[c], v).unsqueeze(0)
                        self.recording[c][v] = torch.cat(
                            (self.recording[c][v][1:].type(data.type()), data), 0
                        )

            self.i += 1

    def save(self, path: str, fmt: str = "npz") -> None:
        # language=rst
        """
        Write the recording dictionary out to file.

        :param path: The directory to which to write the monitor's recording.
        :param fmt: Type of file to write to disk. One of ``"pickle"`` or ``"npz"``.
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        if fmt == "npz":
            # Build a list of arrays to write to disk.
            arrays = {}
            for o in self.recording:
                if type(o) == tuple:
                    arrays.update(
                        {
                            "_".join(["-".join(o), v]): self.recording[o][v]
                            for v in self.recording[o]
                        }
                    )
                elif type(o) == str:
                    arrays.update(
                        {
                            "_".join([o, v]): self.recording[o][v]
                            for v in self.recording[o]
                        }
                    )

            np.savez_compressed(path, **arrays)

        elif fmt == "pickle":
            with open(path, "wb") as f:
                torch.save(self.recording, f)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets recordings to empty ``torch.Tensors``.
        """
        # Reset to empty recordings
        self.recording = {k: {} for k in self.layers + self.connections}

        if self.time is not None:
            self.i = 0

        # If no simulation time is specified, specify 0-dimensional recordings.
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.Tensor()

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.Tensor()

        # If simulation time is specified, pre-allocate recordings in memory for speed.
        else:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.zeros(
                            self.time, *getattr(self.network.layers[l], v).size()
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.zeros(
                            self.time, *getattr(self.network.layers[c], v).size()
                        )
 
class TensorBoardMonitor(AbstractMonitor):
    def __init__(
        self,
        network: "Network",
        state_vars: Iterable[str] = None,
        layers: Optional[Iterable[str]] = None,
        connections: Optional[Iterable[str]] = None,
        time: Optional[int] = None,
        **kwargs,
        ) -> None:
        """
        Constructs a ``TensorBoard`` callback.

        :param network: Network to record state variables from.
        :param layers: Layers to record state variables from.
        :param connections: Connections to record state variables from.
        :param state_vars: List of strings indicating names of state variables to record.
        :param rewards: whether to record rewards.

        Keyword arguments:

        :param str log_dir: Save directory location. Default is runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each
        run. Use hierarchical folder structure to compare between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2',
        etc. for each new experiment to compare across them.
        :param string comment: Comment log_dir suffix appended to the default log_dir. If log_dir is assigned, this argument 
        has no effect.
        :param int purge_step: When logging crashes at step T+X and restarts at step T, any events whose global_step larger 
        or equal to T will be purged and hidden from TensorBoard. Note that crashed and resumed experiments should have 
        the same log_dir.
        :param int max_queue: Size of the queue for pending events and summaries before one of the 'add' calls forces a flush
        to disk. Default is ten items.
        :param int flush_secs: How often, in seconds, to flush the pending events and summaries to disk. Default is every two
        minutes.
        :param string filename_suffix: Suffix added to all event filenames in the log_dir directory. More details on filename
        construction in tensorboard.summary.writer.event_file_writer.EventFileWriter.
        """
        # Initialize tensorboard SummaryWriter object.
        self.writer = SummaryWriter(**kwargs)
        self.step = 0

        # Initialize network, layers, and connections.
        self.network = network
        self.layers = layers if layers is not None else list(self.network.layers.keys())
        self.connections = (
            connections
            if connections is not None
            else list(self.network.connections.keys())
        )
        self.state_vars = state_vars if state_vars is not None else ("v", "s")
        self.time = time

        if self.time is not None:
            self.i = 0

        # Initialize empty recording.
        self.recording = {k: {} for k in self.layers + self.connections}

        # If no simulation time is specified, specify 0-dimensional recordings.
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.Tensor()

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.Tensor()

        # If simulation time is specified, pre-allocate recordings in memory for speed.
        else:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.zeros(
                            self.time, *getattr(self.network.layers[l], v).size()
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.zeros(
                            self.time, *getattr(self.network.connections[c], v).size()
                        )

        # use tags to map the network parameters names to readable names
        self.tags ={
            's': 'Spikes',
            'v': 'Voltages',
            'x': 'Eligibility trace'            
        }

    def record(self, **kwargs) -> None:
        # language=rst
        """
        Appends the current value of the recorded state variables to the recording.
        """
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        data = getattr(self.network.layers[l], v).unsqueeze(0).float()
                        self.recording[l][v] = torch.cat(
                            (self.recording[l][v], data), 0
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        data = getattr(self.network.connections[c], v).unsqueeze(0)
                        self.recording[c][v] = torch.cat(
                            (self.recording[c][v], data), 0
                        )

        else:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        data = getattr(self.network.layers[l], v).float().unsqueeze(0)
                        self.recording[l][v] = torch.cat(
                            (self.recording[l][v][1:].type(data.type()), data), 0
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        data = getattr(self.network.connections[c], v).unsqueeze(0)
                        self.recording[c][v] = torch.cat(
                            (self.recording[c][v][1:].type(data.type()), data), 0
                        )

            self.i += 1

        if kwargs.get('reward', None):
            if self.recording.get('reward', None) is None:
                self.recording['reward'] = []
            self.recording['reward'].append(kwargs['reward'])
            

    def _add_weights(self):
        # language=rst
        """
        Add weights histograms to the SummeryWriter.
        """
        for c in self.connections:
            if hasattr(c, 'mask'):
                self.writer.add_histogram(
                    f'{c[0]} to {c[1]}/Weights',
                    (self.network.connections[c].w)[c.mask.logical_not()].clone(),
                    self.step
                    )
            else:
                self.writer.add_histogram(
                    f'{c[0]} to {c[1]}/Weights',
                    self.network.connections[c].w.clone(),
                    self.step
                    )
            if (
                self.network.connections[c].b is not None 
                and self.network.connections[c].b.any()
            ):
                self.writer.add_histogram(
                    f'{c[0]} to {c[1]}/Biases',
                    self.network.connections[c].b.clone(),
                    self.step
                    )

    def _add_scalers(self):
        # language=rst
        """
        Add state variables plots to the SummeryWriter.
        """
        for v in self.state_vars:
            for l in self.layers:
                if hasattr(self.network.layers[l], v):
                    self.writer.add_scalar(
                        l + '/' + self.tags.get(v, v) + ' (mean)',
                        self.recording[l][v].mean(),
                        self.step
                        )
                
            for c in self.connections:
                if hasattr(self.network.connections[c], v):
                    self.writer.add_scalar(
                        c[0] + ' to ' + c[1] + '/' + self.tags.get(v, v) + ' (mean)',
                        self.recording[c][v].mean(),
                        self.step
                        )

    def _add_grids(self):
        # language=rst
        """
        Add state variables grids to the SummeryWriter.
        """
        for v in self.state_vars:
            for l in self.layers:
                if hasattr(self.network.layers[l], v):
                    # Shuffle variable into 1x1x#neuronsxT
                    grid = self.recording[l][v].view(self.recording[l][v].shape[0], -1)
                    self.writer.add_image(
                        l + '/' + self.tags.get(v, v) + ' grid',
                        grid,
                        self.step,
                        dataformats= 'HW',
                        )
                
            for c in self.connections:
                if hasattr(self.network.connections[c], v):
                    # Shuffle variable into 1x1x#neuronsxT
                    grid = self.recording[c][v].view(self.recording[c][v].shape[0], -1)
                    self.writer.add_image(
                        c[0] + ' to ' + c[1] + '/' + self.tags.get(v, v) + ' grid',
                        grid, 
                        self.step,
                        dataformats= 'HW',
                        )

    def update(self, step = None) -> None:
        # language=rst
        """
        Adds data to tensorboard after every step.
        """
        if step: 
            self.step = step
        self._add_weights()
        self._add_scalers()
        self._add_grids()
        self.step += 1

        self.writer.flush()

    #TODO
    def plot_reward(
        self,
        reward_list: list,
        reward_window: int = None,
        tag: str = "reward",
        step: int = None,
    ) -> None:
        # language=rst
        """
        Plot the accumulated reward for each episode.

        :param reward_list: The list of recent rewards to be plotted.
        :param reward_window: The length of the window to compute a moving average over.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        self.writer.add_scalar(tag, reward_list[-1], step)

    def plot_obs(self, obs: torch.Tensor, tag: str = "obs", step: int = None) -> None:
        # language=rst
        """
        Pulls the observation off of torch and sets up for Matplotlib
        plotting.

        :param obs: A 2D array of floats depicting an input image.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        obs_grid = make_grid(obs.float(), nrow=4, normalize=True)
        self.writer.add_image(tag, obs_grid, step)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets recordings to empty ``torch.Tensors``.
        """
        # Reset to empty recordings
        self.recording = {k: {} for k in self.layers + self.connections}

        if self.time is not None:
            self.i = 0

        # If no simulation time is specified, specify 0-dimensional recordings.
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.Tensor()

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.Tensor()

        # If simulation time is specified, pre-allocate recordings in memory for speed.
        else:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.zeros(
                            self.time, *getattr(self.network.layers[l], v).size()
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.zeros(
                            self.time, *getattr(self.network.layers[c], v).size()
                        )

        # Reset rewards 
        if self.recording.get('reward', None) is not None:
            self.recording['reward'] = []
