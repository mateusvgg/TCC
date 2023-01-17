import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Generator

from src.projections_dataclasses import PlyProjections, PairProjections


def prepare_ref_image(
    image_path: str,
    resize: bool = True
) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    if resize and (min(image.size) > 256):
        image = transforms.functional.resize(image, 256)  # type: ignore
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0), original_size


def prepare_deg_image(image_path: str, new_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(new_size)
    if min(image.size) > 256:
        image = transforms.functional.resize(image, 256)  # type: ignore
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

    def _get_refs_projections(self) -> Dict[str, tuple[PlyProjections, tuple[int, int]]]:
        ''' From the dataset with the data mapping in disk, generate 
        the PlyProjections for every reference PC. '''

        refs_names = list(self.df_data['REF'].unique())
        refs_projections = {}
        for ref_ in refs_names:
            ref_name = ref_.replace(".ply", '')
            views_paths = self._gen_views_paths(ref_name, self.ref_base)
            ref_images = [prepare_ref_image(view_path)
                          for view_path in views_paths]
            refs_projections_aux = [ref_image[0] for ref_image in ref_images]
            sizes = [ref_image[1] for ref_image in ref_images]
            projs = self._gen_projections_obj(ref_name, refs_projections_aux)
            refs_projections[ref_name] = (projs, sizes)
        return refs_projections

    def _get_pairs_projections(
        self,
        refs_projections: Dict[str, tuple[PlyProjections, tuple[int, int]]]
    ) -> List[PairProjections]:

        projections = []
        for _, row in tqdm(self.df_data.iterrows()):
            ref_name = row['REF'].replace(".ply", '').strip()
            deg_name = row['SIGNAL'].replace(".ply", '').strip()

            views_paths = self._gen_views_paths(deg_name, self.deg_base)

            ref_shapes = [shape for shape in refs_projections[ref_name][1]]
            degs_projections_aux = [
                prepare_deg_image(view_path, ref_shape)  # type: ignore
                for view_path, ref_shape in zip(views_paths, ref_shapes)
            ]

            projs = self._gen_projections_obj(deg_name, degs_projections_aux)
            projs_pair = PairProjections(
                ref=refs_projections[ref_name][0],
                deg=projs,
                score=row['SCORE']
            )
            projections.append(projs_pair)
        return projections

    def data_generator(self) -> Generator[PairProjections, None, None]:
        refs_projections = self._get_refs_projections()

        for _, row in self.df_data.iterrows():
            ref_name = row['REF'].replace(".ply", '').strip()
            deg_name = row['SIGNAL'].replace(".ply", '').strip()

            views_paths = self._gen_views_paths(deg_name, self.deg_base)

            ref_shapes = [shape for shape in refs_projections[ref_name][1]]
            degs_projections_aux = [
                prepare_deg_image(view_path, ref_shape)  # type: ignore
                for view_path, ref_shape in zip(views_paths, ref_shapes)
            ]

            projs = self._gen_projections_obj(deg_name, degs_projections_aux)
            projs_pair = PairProjections(
                ref=refs_projections[ref_name][0],
                deg=projs,
                score=row['SCORE']
            )
            yield projs_pair

    def prepare_data(self) -> List[PairProjections]:
        refs_projections = self._get_refs_projections()
        pair_projections = self._get_pairs_projections(refs_projections)
        return pair_projections
