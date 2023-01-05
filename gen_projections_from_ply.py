import numpy as np
import open3d as o3d
import cv2

import ProjQM


def gen_projections(path_to_pc: str, path_to_save: str) -> None:
    pc = o3d.io.read_point_cloud(path_to_pc)
    ImlistR, OMListR = ProjQM.orthographic_projection(pc, 10, 2)

    for i in range(len(ImlistR)):
        ImR_curr = ImlistR[i].astype(np.uint8)
        OMR_curr = OMListR[i]

        ImR_cropped, OMR_cropped = ProjQM.cropp_images(ImR_curr, OMR_curr)

        ImR_padded = ProjQM.pad_images(ImR_cropped, OMR_cropped)
        ProjQM.save_images(f'{path_to_save}_view_{i}.bmp', cv2.cvtColor(ImR_padded, cv2.COLOR_BGR2RGB))