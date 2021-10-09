from abc import ABC, abstractmethod
from typing import Union, Optional, Iterable, Dict

import torch

class CallbackList:
    def __init__(
        self,
        callbacks
        ) -> None:
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks
        for callback in callbacks:
            assert isinstance(callback,Callback), (
                "All elements must be an instance of 'Callback' class. " +
                f"Found {callback} of type {type(callback)}"
                )

    def __iter__(self) -> None:
        for elem in self.callbacks:
            yield elem

    def set_network(self, network) -> None:
        if self.callbacks:
            for callback in self.callbacks:
                callback.set_network(network)

    def on_run_start(self, kwargs) -> None:
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_run_start(kwargs)

    def on_run_end(self, kwargs) -> None:
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_run_end(kwargs)

    def on_timepoint_start(self, timepoint, kwargs) -> None:
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_timepoint_start(timepoint, kwargs)
            
    def on_timepoint_end(self, timepoint, kwargs) -> None:
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_timepoint_end(timepoint, kwargs)    


class Callback:

    @abstractmethod
    def set_network(self, network) -> None:
        self.network = network

    @abstractmethod
    def on_run_start(self, kwargs) -> None:
        ...

    @abstractmethod
    def on_run_end(self, kwargs) -> None:
        ...

    @abstractmethod
    def on_timepoint_start(self, timepoint, kwargs) -> None:
        ...

    @abstractmethod
    def on_timepoint_end(self, timepoint, kwargs) -> None:
        ...


class TensorBoard(Callback):
    def __init__(self,
        state_vars: Iterable[str] = None,
        layers: Optional[Iterable[str]] = None,
        connections: Optional[Iterable[str]] = None,
        log_dir: str = None,
        rewards: Optional[bool] = False,
        ) -> None:
        """
        Constructs a ``TensorBoard`` callback.

        :param layers: Layers to record state variables from.
        :param connections: Connections to record state variables from.
        :param state_vars: List of strings indicating names of state variables to record.
        :param log_dir: Save directory location. Default is runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each
        run. Use hierarchical folder structure to compare between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2',
        etc. for each new experiment to compare across them.
        :param rewards: whether to record rewards.
        """
        super().__init__()

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir =log_dir)

        self.layers_names = layers
        self.connections_names = connections

        self.state_vars = state_vars if state_vars is not None else ("v", "s")
        self.rewards = rewards
        self.n_runs = 0

    def set_network(self, network) -> None:
        super().set_network(network)

        # self.writer.add_graph(network)

        self.layers = (
            self.layers_names 
            if self.layers_names is not None 
            else list(self.network.layers.keys())
        )
        self.connections = (
            self.connections_names
            if self.connections_names is not None
            else list(self.network.connections.keys())
        )

        # Initialize empty recording.
        self.recording = {k: {} for k in self.layers + self.connections}
        if self.rewards:
            self.recording['rewards'] = torch.Tensor()
        
    def on_run_start(self, kwargs) -> None:
        for v in self.state_vars:
            for l in self.layers:
                if hasattr(self.network.layers[l], v):
                    self.recording[l][v] = torch.Tensor()

            for c in self.connections:
                if hasattr(self.network.connections[c], v):
                    self.recording[c][v] = torch.Tensor()

        if self.n_runs ==0:
            for c in self.connections:
                self.writer.add_histogram(
                    f'{c[0]} to {c[1]}/Weights',
                    self.network.connections[c].w,
                    self.n_runs
                    )
                if self.network.connections[c].b is not None:
                    self.writer.add_histogram(
                        f'{c[0]} to {c[1]}/Biases',
                        self.network.connections[c].w,
                        self.n_runs
                        )

    def on_run_end(self, kwargs) -> None:
        self.n_runs += 1

        for v in self.state_vars:
            for l in self.layers:
                if hasattr(self.network.layers[l], v):
                    self.writer.add_scalar(
                        l + '/' + v,
                        self.recording[l][v].sum(),
                        self.n_runs
                        )
                
            for c in self.connections:
                if hasattr(self.network.connections[c], v):
                    self.writer.add_scalar(
                        c[0] + ' to ' + c[1] + '/' + v,
                        self.recording[c][v].sum(),
                        self.n_runs
                        )

        for c in self.connections:
            self.writer.add_histogram(
                f'{c[0]} to {c[1]}/Weights',
                self.network.connections[c].w,
                self.n_runs
                )
            if self.network.connections[c].b is not None:
                self.writer.add_histogram(
                    f'{c[0]} to {c[1]}/Biases',
                    self.network.connections[c].w,
                    self.n_runs
                    )

        if self.rewards:
            self.writer.add_scalar(
                'Rewards',
                self.recording['rewards'].sum(),
                self.n_runs
                )

    def on_timepoint_end(self, timepoint, kwargs) -> None:
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

        if self.rewards:
            data = torch.tensor([kwargs['reward']])
            self.recording['rewards'] = torch.cat(
                        (self.recording['rewards'], data), 0
                    )

class DopaminergicNeurons(Callback):
    # language=rst
    """
    Release dopamine (reward) into network based on neuronal activity of dopaminergic layers 
    """
    def __init__(
        self,
        layers: Dict[str, float]) -> None:
        # language=rst
        """
        Constructor for dopaminergic neurons reward function.

        :params layers: Dictionary of dopaminergic layers along with their 'DA per spike' release rate.
        """
        self.layers = layers 

    def on_timepoint_start(self, timepoint, kwargs) -> None:
        # language=rst
        """
        Compute the amount of released dopamine (DA) as reward.
        """
        for l in self.layers:
            kwargs['reward'] = self.network.layers[l].s.sum() * self.layers[l]