from tqdm import tqdm
from PIL import Image
import torch
from typing import List
from mesh_reconstruction.remesh import calc_vertex_normals
from mesh_reconstruction.opt import MeshOptimizer
from mesh_reconstruction.func import make_star_cameras_orthographic, make_star_cameras_orthographic_py3d
from mesh_reconstruction.render import NormalsRenderer, Pytorch3DNormalsRenderer
from scripts.project_mesh import multiview_color_projection, get_cameras_list, get_sparse_cameras_list
from scripts.utils import to_py3d_mesh, from_py3d_mesh, init_target
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
def run_mesh_refine(vertices, faces, pils: List[Image.Image], sparse_cameras, steps=100, start_edge_len=0.02, end_edge_len=0.005, decay=0.99, update_normal_interval=10, update_warmup=10, return_mesh=True, process_inputs=True, process_outputs=True):
    if process_inputs:
        vertices = vertices * 2 / 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]
    
    poission_steps = []
    to_pil = torchvision.transforms.ToPILImage()
    # assert len(pils) == 4 # normal
    # mv,proj = make_star_cameras_orthographic(4, 1)      
    # TODO： front back  
    # mv = mv[[0,2]]
    renderer = NormalsRenderer(mv=None,proj=None,image_size=list(pils[0].size),sparse_cameras=sparse_cameras)
    # cameras = make_star_cameras_orthographic_py3d([0, 270, 180, 90], device="cuda", focal=1., dist=4.0)
    # renderer = Pytorch3DNormalsRenderer(cameras, list(pils[0].size), device="cuda")
    target_images = init_target(pils, new_bkgd=(0., 0., 0.)) # 4s
    # 1. no rotate
    # TODO： front back  
    # sparse_target_images = (torch.stack([torch.from_numpy(np.array(img, dtype=np.float32)) for img in pils[2:]]) / 255).to(device=target_images.device)
    # target_images = torch.cat([target_images[:2], sparse_target_images], dim=0)
    
    # 2. init from coarse mesh
    opt = MeshOptimizer(vertices,faces, ramp=5, edge_len_lims=(end_edge_len, start_edge_len), local_edgelen=False, laplacian_weight=0.02)

    vertices = opt.vertices
    alpha_init = None
    
    mask = target_images[..., -1] < 0.5

    normals = calc_vertex_normals(vertices,faces)
    images = renderer.render(vertices,normals,faces) # normal
        # TODO: resize to input size, d_mask need change 
        # print(images.shape)

    # to_pil(images[2].permute(2,0,1)).convert('RGB').save('render2.jpg')
    # to_pil(images[3].permute(2,0,1)).convert('RGB').save('render3.jpg')
    # to_pil(images[6].permute(2,0,1)).convert('RGB').save('render2.jpg')
    
    for i, img in enumerate(target_images):
        to_pil(img.permute(2,0,1)).convert('RGB').save(f'target_images{i}.jpg')
        alpha_channel = img[:, :, 3]
        alpha_pil = to_pil(alpha_channel).convert('L')  # 'L'模式表示灰度图像
        alpha_pil.save(f'target_images{i}_alpha.jpg')
    


    
    for i in tqdm(range(steps)):
        opt.zero_grad()
        opt._lr *= decay
        normals = calc_vertex_normals(vertices,faces)
        images = renderer.render(vertices,normals,faces) # normal
        # TODO: resize to input size, d_mask need change 
        # print(images.shape)

        for j, img in enumerate(images):
            to_pil(img.permute(2,0,1)).convert('RGB').save(f'render{j}.jpg')
            alpha_channel = img[:, :, 3]
            alpha_pil = to_pil(alpha_channel).convert('L')  # 'L'模式表示灰度图像
            alpha_pil.save(f'render{j}_alpha.jpg')
        
        if alpha_init is None:
            alpha_init = images.detach()
        
        if i < update_warmup or i % update_normal_interval == 0:
            with torch.no_grad():
                py3d_mesh = to_py3d_mesh(vertices, faces, normals)
                # cameras = get_cameras_list(azim_list = [0, 90, 180, 270], device=vertices.device, focal=1.)
                # cameras = get_cameras_list(azim_list = [0, 180], device=vertices.device, focal=1.)
                cameras = sparse_cameras
                
                import matplotlib.pyplot as plt
                def plot_pose(c2ws, cache_dir, name):
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    for i, c2w in enumerate(c2ws):
                        R = c2w[:3, :3]
                        t = c2w[:3, 3]
                        ax.scatter(t[0], t[1], t[2], color='r')
                        for j, color in enumerate(['r', 'g', 'b']):
                            end = t + R.T[j] * 1
                            ax.plot([t[0], end[0]], [t[1], end[1]], [t[2], end[2]], color=color)
                        direction = -R[2]  # Assuming the camera is looking along the negative z-axis
                        # ax.quiver(t[0], t[1], t[2], direction[0], direction[1], direction[2], length=0.1, color='y')

                        ax.text(t[0], t[1], t[2], str(i), color='k')

                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    plt.gca().view_init(elev=90, azim=0)
                    plt.show()
                    fig.savefig(f'./pose_'+name+'_.png')

                    plt.gca().view_init(elev=30, azim=30)
                    plt.show()
                    fig.savefig(f'./pose1_'+name+'_.png')

                    plt.gca().view_init(elev=0, azim=30)
                    plt.show()
                    fig.savefig(f'./pose2_'+name+'_.png')

                c2ws = []
                for camera in cameras:
                    w2c = camera.get_world_to_view_transform(R=camera.R, T=camera.T).get_matrix().float()
                    w2c = w2c.T.reshape(4,4)
                    c2w = torch.inverse(w2c)
                    c2ws.append(c2w)
                c2ws = np.array([c2w.cpu().numpy() for c2w in c2ws])
                print(c2ws.shape)
                cache_dir = "./" 
                name = "camera_poses"
                plot_pose(c2ws, cache_dir, name)
                
                # pils = pils[:4] #TODO
                # cameras = cameras[:4] #TODO
                from pytorch3d.io import save_obj
                
                # from scripts.mesh_filter import clean_mesh_connected
                # verts_lists = []
                # faces_lists = []
                # for verts, faces in zip(py3d_mesh.verts_list(), py3d_mesh.faces_list()):
                #     verts, faces  = clean_mesh_connected(verts, faces)
                #     verts_lists.append(verts)
                #     faces_lists.append(faces)
                # py3d_mesh._verts_list = verts_lists
                # py3d_mesh._faces_list = faces_lists

                verts = py3d_mesh.verts_list()[0]
                faces = py3d_mesh.faces_list()[0]
                save_obj("refine.obj", verts, faces)
                _, _, target_normal = from_py3d_mesh(multiview_color_projection(py3d_mesh, pils, cameras_list=cameras, 
                                                                                weights=[1.0] + [1.0] * (len(cameras)-1),
                                                                                # weights=[2.0, 0.8, 1.0, 0.8] + [2.0] * (len(cameras)-4), 
                                                                                confidence_threshold=0.1, complete_unseen=False, below_confidence_strategy='original', reweight_with_cosangle='linear'))
                # t = pils[0].convert("RGB")
                # t.save("0.jpg")
                # t = pils[1].convert("RGB")
                # t.save("1.jpg")
                

                target_normal = target_normal * 2 - 1
                target_normal = torch.nn.functional.normalize(target_normal, dim=-1)
                debug_images = renderer.render(vertices,target_normal,faces)
                # TODO: resize to input size
                to_pil(debug_images[2].permute(2,0,1)).convert('RGB').save('debug_images[0].jpg')
                print(target_normal.shape, debug_images.shape)
    
        # images = images[:4]
        # debug_images = debug_images[:4]
        # target_images = target_images[:4]
        # mask = mask[:4]

        d_mask = images[..., -1] > 0.5
        loss_debug_l2 = (images[..., :3][d_mask] - debug_images[..., :3][d_mask]).pow(2).mean()
        loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()
        
        loss = loss_debug_l2 + loss_alpha_target_mask_l2
        # loss = loss_alpha_target_mask_l2
        
        # out of box
        loss_oob = (vertices.abs() > 0.99).float().mean() * 10
        loss = loss + loss_oob
        
        loss.backward()
        opt.step()
        
        vertices,faces = opt.remesh(poisson=(i in poission_steps))
    
    vertices, faces = vertices.detach(), faces.detach()
    
    if process_outputs:
        vertices = vertices / 2 * 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]

    if return_mesh:
        return to_py3d_mesh(vertices, faces)
    else:
        return vertices, faces
