from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Iterable, Optional, Union, Sequence, Type

import numpy as np
import torch
from torch.nn import Module, Parameter

from .nodes import Nodes
from .topology import AbstractConnection, Connection
from bindsnet import manual_seed

class MaskedConnection(Connection):
    """ """
    def __init__(
        self,
        nodes: Nodes,
        connection_chance: float,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ):
        """
        [summary]

        """
        super().__init__(nodes, nodes, nu, reduction, weight_decay, **kwargs)
        mask = torch.bernoulli(
            connection_chance * torch.ones(*nodes.shape, *nodes.shape)
        )
        self.register_buffer("mask", mask)
        self.w.masked_fill_(mask, 0)

class ConnectedNodes(ABC, Module):
    # language=rst
    """
    [summary]
    """

    def __init__(
        self,
        nodes_type: Type[Nodes],
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        nu: Optional[Union[float, Sequence[float]]] = None,
        **kwargs
    ) -> None:
        # language=rst
        """
        [summary]
        """
        super().__init__()
        self.nodes_type = nodes_type
        self.n = n
        self.shape = shape

        b = kwargs.get("b", None)
        if b is not None:
            self.b = Parameter(b, requires_grad=False)
        else:
            self.b = None


class RandomConnections(ConnectedNodes):
    def __init__(
        self,
        nodes_type: Type[Nodes],
        connection_chance: float, 
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        exc_rate: float = 0.8,
        nu: Optional[Union[float, Sequence[float]]] = None,
        **kwargs
    ) -> None:
        # language=rst
        """
        [summary]
        """
        super().__init__(nodes_type, n, shape, nu, **kwargs)

        wmin = kwargs.get("wmin", -np.inf)
        wmax = kwargs.get("wmax", np.inf)
        assert wmin<0 and wmax>0, '' #TODO

        w = torch.rand(self.n, self.n)
        w[np.diag_indices(w.shape[0])] = 0.

        n_exc = int(exc_rate * self.n)
        n_inh = n - n_exc
        exc_inh = torch.tensor([1] * n_exc + [-1] * n_inh)
        w *= exc_inh
    
        self.wmin = torch.tensor([[0] * n] * n_exc + [[wmin] * n] * n_inh)
        self.wmax = torch.tensor([[wmax] * n] * n_exc + [[0] * n] * n_inh)
        w = torch.clamp(w, self.wmin, self.wmax)

        self.nodes = self.nodes_type(
            self.n,
            self.shape,
            kwargs.get("traces", False),
            kwargs.get("traces_additive", False),
            kwargs.get("tc_trace", 20.0),
            kwargs.get("trace_scale", 1.0),
            kwargs.get("sum_input", False),
            kwargs.get("learning", True),
            )

        self.connection = MaskedConnection(
            self.nodes,
            connection_chance,
            nu,
            kwargs.get("reduction", None), 
            kwargs.get("weight_decay", 0.0),
            w = w, 
            wmin = self.wmin,
            wmax = self.wmax,
            )
