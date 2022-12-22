import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
from typing import Dict, List

from projections_dataclasses import PlyProjections, PairProjections


def prepare_image(image_path: str, resize: bool = True) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if resize and (min(image.size) > 256):
        image = transforms.functional.resize(image, 256) # type: ignore
    image = transforms.ToTensor()(image)  
    return image.unsqueeze(0)



class LoadProjectionsData:

    def __init__(self, ref_base: str, deg_base: str, df_data: pd.DataFrame) -> None:
        self.ref_base = ref_base 
        self.deg_base = deg_base 
        self.df_data = df_data


    def _gen_views_paths(self, pc_name: str, base_path: str) -> List[str]:
        ''' From a PC name, generate the paths of the 6 views. '''

        views_names = [f'{pc_name}_view_{i}.bmp' for i in range(6)]
        views_paths = [
            os.path.join(*[base_path, view_name]) for view_name in views_names
        ]
        return views_paths


    def _gen_projections_obj(self, pc_name: str, projections: List[torch.Tensor]) -> PlyProjections:
        projs = PlyProjections(
            name=pc_name,
            view0=projections[0],
            view1=projections[1],
            view2=projections[2],
            view3=projections[3],
            view4=projections[4],
            view5=projections[5]
        )
        return projs


    def _get_refs_projections(self) -> Dict[str, PlyProjections]:
        ''' From the dataset with the data mapping in disk, generate 
        the PlyProjections for every reference PC. '''

        refs_names = list(self.df_data['REF'].unique())
        refs_projections = {}
        for ref_ in refs_names:
            ref_name = ref_.replace(".ply", '')
            views_paths = self._gen_views_paths(ref_name, self.ref_base)
            refs_projections_aux = [prepare_image(view_path) for view_path in views_paths]
            projs = self._gen_projections_obj(ref_name, refs_projections_aux)
            refs_projections[ref_name] = projs
        return refs_projections


    def _get_pairs_projections(
        self,
        refs_projections: Dict[str, PlyProjections]
    ) -> List[PairProjections]:
        
        projections = []
        for _, row in tqdm(self.df_data.iterrows()):
            ref_name = row['REF'].replace(".ply", '').strip()
            deg_name = row['SIGNAL'].replace(".ply", '').strip()

            views_paths = self._gen_views_paths(deg_name, self.deg_base)

            degs_projections_aux = [prepare_image(view_path) for view_path in views_paths]
            projs = self._gen_projections_obj(deg_name, degs_projections_aux)
            projs_pair = PairProjections(
                ref=refs_projections[ref_name],
                deg=projs,
                score=row['SCORE']
            )
            projections.append(projs_pair)
        return projections

    
    def prepare_data(self) -> List[PairProjections]:
        refs_projections = self._get_refs_projections()
        pair_projections = self._get_pairs_projections(refs_projections)
        return pair_projections