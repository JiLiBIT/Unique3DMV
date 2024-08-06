from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from typing import List
from mesh_reconstruction.remesh import calc_vertex_normals
from mesh_reconstruction.opt import MeshOptimizer
from mesh_reconstruction.func import make_star_cameras_orthographic, make_star_cameras_orthographic_py3d
from mesh_reconstruction.render import NormalsRenderer, Pytorch3DNormalsRenderer
from scripts.utils import to_py3d_mesh, init_target
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

def reconstruct_stage1(pils: List[Image.Image], sparse_cameras, steps=100, vertices=None, faces=None, start_edge_len=0.15, end_edge_len=0.005, decay=0.995, return_mesh=True, loss_expansion_weight=0.1, gain=0.1):
    
    # assert len(pils) == 4
    to_pil = torchvision.transforms.ToPILImage()
    mv,proj = make_star_cameras_orthographic(4, 1)
    # TODO： front back  
    mv = mv[[0,2]]
    
    renderer = NormalsRenderer(mv=None,proj=None,image_size=list(pils[0].size),sparse_cameras=sparse_cameras)
    # cameras = make_star_cameras_orthographic_py3d([0, 270, 180, 90], device="cuda", focal=1., dist=4.0)
    # renderer = Pytorch3DNormalsRenderer(cameras, list(pils[0].size), device="cuda")
    
    target_images = init_target(pils, new_bkgd=(0., 0., 0.)) # 4s
    # 1. no rotate
    # TODO： front back  
    # sparse_target_images = (torch.stack([torch.from_numpy(np.array(img, dtype=np.float32)) for img in pils[2:]]) / 255).to(device=target_images.device)
    # target_images = torch.cat([target_images[:2], sparse_target_images], dim=0)

    # 2. init from coarse mesh
    for i, img in enumerate(target_images):
        to_pil(img.permute(2,0,1)).convert('RGB').save(f'target_images{i}.jpg')
        alpha_channel = img[:, :, 3]
        alpha_pil = to_pil(alpha_channel).convert('L')  # 'L'模式表示灰度图像
        alpha_pil.save(f'target_images{i}_alpha.jpg')
    opt = MeshOptimizer(vertices,faces, local_edgelen=False, gain=gain, edge_len_lims=(end_edge_len, start_edge_len))

    vertices = opt.vertices
    normals = calc_vertex_normals(vertices,faces)
    images = renderer.render(vertices,normals,faces)
    for j, img in enumerate(images):
        to_pil(img.permute(2,0,1)).convert('RGB').save(f'init{j}.jpg')
        alpha_channel = img[:, :, 3]
        alpha_pil = to_pil(alpha_channel).convert('L')  # 'L'模式表示灰度图像
        alpha_pil.save(f'init{j}_alpha.jpg')
    mask = target_images[..., -1] < 0.5
    for i in tqdm(range(steps)):
        opt.zero_grad()
        opt._lr *= decay
        normals = calc_vertex_normals(vertices,faces)
        images = renderer.render(vertices,normals,faces)
        for j, img in enumerate(images):
            to_pil(img.permute(2,0,1)).convert('RGB').save(f'render{j}.jpg')
            alpha_channel = img[:, :, 3]
            alpha_pil = to_pil(alpha_channel).convert('L')  # 'L'模式表示灰度图像
            alpha_pil.save(f'render{j}_alpha.jpg')
        loss_expand = 0.5 * ((vertices+normals).detach() - vertices).pow(2).mean()
        
        t_mask = images[..., -1] > 0.5
        loss_target_l2 = (images[t_mask] - target_images[t_mask]).abs().pow(2).mean()
        loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()

        loss = loss_target_l2 + loss_alpha_target_mask_l2 + loss_expand * loss_expansion_weight
        
        # out of box
        loss_oob = (vertices.abs() > 0.99).float().mean() * 10
        loss = loss + loss_oob
        
        loss.backward()
        opt.step()

        vertices,faces = opt.remesh(poisson=False)

    vertices, faces = vertices.detach(), faces.detach()
    
    if return_mesh:
        return to_py3d_mesh(vertices, faces)
    else:
        return vertices, faces
