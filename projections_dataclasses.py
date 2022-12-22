import torch
from dataclasses import dataclass


@dataclass
class PlyProjections:
    name: str
    view0: torch.Tensor
    view1: torch.Tensor
    view2: torch.Tensor
    view3: torch.Tensor
    view4: torch.Tensor
    view5: torch.Tensor

    def __iter__(self):
        views = [self.view0, self.view1, self.view2, self.view3, self.view4, self.view5]
        for view in views:
            yield view
        
        
@dataclass
class PairProjections:
    ref: PlyProjections
    deg: PlyProjections
    score: float