from typing import List
import torch
import numpy as np
from PIL import Image
from pytorch3d.renderer.cameras import look_at_view_transform, OrthographicCameras, PerspectiveCameras, CamerasBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    TexturesVertex,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
)
from pytorch3d.renderer import MeshRasterizer
import nvdiffrast.torch as dr
import os 
import sys
current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_folder}')
from colmap_utils import read_cameras_binary, read_images_binary
import math

def get_camera(world_to_cam, fov_in_degrees=60, focal_length=1 / (2**0.5), cam_type='fov', K=None):
    # pytorch3d expects transforms as row-vectors, so flip rotation: https://github.com/facebookresearch/pytorch3d/issues/1183
    R = world_to_cam[:3, :3].t()[None, ...] #
    T = world_to_cam[:3, 3][None, ...]
    if cam_type == 'fov':
        if K is not None:
            camera = FoVPerspectiveCameras(device=world_to_cam.device, R=R, T=T, K=K, degrees=True)
        else:
            camera = FoVPerspectiveCameras(device=world_to_cam.device, R=R, T=T,
                                                fov=65, degrees=True,
                                                znear=0.1, zfar=50.0, aspect_ratio=1.0)
    else:
        focal_length = 1 / focal_length
        camera = FoVOrthographicCameras(device=world_to_cam.device, R=R, T=T, min_x=-focal_length, max_x=focal_length, min_y=-focal_length, max_y=focal_length)
    return camera

def render_pix2faces_py3d(meshes, cameras, H=512, W=512, blur_radius=0.0, faces_per_pixel=1):
    """
    Renders pix2face of visible faces.

    :param mesh: Pytorch3d.structures.Meshes
    :param cameras: pytorch3d.renderer.Cameras
    :param H: target image height
    :param W: target image width
    :param blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
    :param faces_per_pixel: (int) Number of faces to keep track of per pixel.
            We return the nearest faces_per_pixel faces along the z-axis.
    """
    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel
    )
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    fragments: Fragments = rasterizer(meshes, cameras=cameras)
    return {
        "pix_to_face": fragments.pix_to_face[..., 0],
    }


def _warmup(glctx, device=None):
    device = 'cuda' if device is None else device
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device=device, **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

class Pix2FacesRenderer:
    def __init__(self, device="cuda"):
        self._glctx = dr.RasterizeGLContext(output_db=False, device=device)
        self.device = device
        _warmup(self._glctx, device)

    def transform_vertices(self, meshes: Meshes, cameras: CamerasBase):
        vertices = cameras.transform_points_ndc(meshes.verts_padded())
        print("vertices:",vertices.shape)
        perspective_correct = cameras.is_perspective()
        znear = cameras.get_znear()
        if isinstance(znear, torch.Tensor):
            znear = znear.min().item()
        z_clip = None if not perspective_correct or znear is None else znear / 2
       
        
        if not cameras.is_perspective():
            vertices = vertices * torch.tensor([1, 1, 1]).to(vertices) # -1, -1, 1
            vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1).to(torch.float32)
        else:
            vertices = vertices * torch.tensor([1, 1, 1]).to(vertices)
            vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1).to(torch.float32)
        # if z_clip: #TODO
        #     vertices = vertices[vertices[..., 2] >= cameras.get_znear()][None]    # clip

        print(cameras.is_perspective(),vertices,vertices[..., 2].mean())
        # False tensor([[[-0.1265,  0.3504,  0.0396,  0.9604],
        # True tensor([[[-0.6371,  0.1615,  3.5801,  4.4011],

        return vertices

    def render_pix2faces_nvdiff(self, meshes: Meshes, cameras: CamerasBase, H=512, W=512):
        meshes = meshes.to(self.device)
        cameras = cameras.to(self.device)
        vertices = self.transform_vertices(meshes, cameras)
        faces = meshes.faces_packed().to(torch.int32)
        from pytorch3d.io import save_obj
        print("vertices: ", vertices.shape, faces.shape)
        save_obj("project_mesh.obj", vertices.squeeze(0)[...,:3], faces)
        
        rast_out,_ = dr.rasterize(self._glctx, vertices, faces, resolution=(H, W), grad_db=False) #C,H,W,4
        pix_to_face = rast_out[..., -1].to(torch.int32) - 1
        # print("rast_out:",rast_out[..., -1])
        import imageio
        rast_out_image = (rast_out * 255).to(torch.uint8).cpu()
        image_data = rast_out_image[0]
        imageio.imwrite('rast_out.png', image_data)
        pix_to_face_image = ((pix_to_face+1)/2 * 255).to(torch.uint8).cpu()
        image_data = pix_to_face_image[0]
        imageio.imwrite('pix_to_face_out.png', image_data)

        return pix_to_face

pix2faces_renderer = Pix2FacesRenderer()

def get_visible_faces(meshes: Meshes, cameras: CamerasBase, resolution=1024):
    # pix_to_face = render_pix2faces_py3d(meshes, cameras, H=resolution, W=resolution)['pix_to_face']
    pix_to_face = pix2faces_renderer.render_pix2faces_nvdiff(meshes, cameras, H=resolution, W=resolution)

    unique_faces = torch.unique(pix_to_face.flatten())
    unique_faces = unique_faces[unique_faces != -1]
    return unique_faces

def project_color(meshes: Meshes, cameras: CamerasBase, pil_image: Image.Image, use_alpha=True, eps=0.05, resolution=1024, device="cuda") -> dict:
    """
    Projects color from a given image onto a 3D mesh.

    Args:
        meshes (pytorch3d.structures.Meshes): The 3D mesh object.
        cameras (pytorch3d.renderer.cameras.CamerasBase): The camera object.
        pil_image (PIL.Image.Image): The input image.
        use_alpha (bool, optional): Whether to use the alpha channel of the image. Defaults to True.
        eps (float, optional): The threshold for selecting visible faces. Defaults to 0.05.
        resolution (int, optional): The resolution of the projection. Defaults to 1024.
        device (str, optional): The device to use for computation. Defaults to "cuda".
        debug (bool, optional): Whether to save debug images. Defaults to False.

    Returns:
        dict: A dictionary containing the following keys:
            - "new_texture" (TexturesVertex): The updated texture with interpolated colors.
            - "valid_verts" (Tensor of [M,3]): The indices of the vertices being projected.
            - "valid_colors" (Tensor of [M,3]): The interpolated colors for the valid vertices.
    """
    meshes = meshes.to(device)
    cameras = cameras.to(device)
    image = torch.from_numpy(np.array(pil_image.convert("RGBA")) / 255.).permute((2, 0, 1)).float().to(device)     # in CHW format of [0, 1.]
    unique_faces = get_visible_faces(meshes, cameras, resolution=resolution) #TODO: resolution=resolution

    
    # visible faces
    faces_normals = meshes.faces_normals_packed()[unique_faces]
    faces_normals = faces_normals / faces_normals.norm(dim=1, keepdim=True)
    world_points = cameras.unproject_points(torch.tensor([[[0., 0., 0.1], [0., 0., 0.2]]]).to(device))[0]
    view_direction = world_points[1] - world_points[0]
    # view_direction = view_direction + translation_vector
    view_direction = view_direction / view_direction.norm(dim=0, keepdim=True)


    verts = meshes.verts_packed()
    faces = meshes.faces_packed()

    face_centers = verts[faces].mean(dim=1)


    def save_vectors_as_obj(vectors, filename, start_points=[[0,0,0]], scale=0.1):
        with open(filename, 'w') as file:
            point_index = 1
            for start_point, vec in zip(start_points, vectors):
                file.write(f"v {start_point[0]} {start_point[1]} {start_point[2]}\n")
                end_point = [start_point[j] + scale * vec[j].item() for j in range(3)]
                file.write(f"v {end_point[0]} {end_point[1]} {end_point[2]}\n")
                file.write(f"l {point_index} {point_index + 1}\n")
                point_index += 2
    # save_vectors_as_obj(torch.stack([view_direction]), 'view_direction2.obj',scale=1.0)
    # save_vectors_as_obj(faces_normals, 'faces_normals.obj', face_centers)


    # find invalid faces
    cos_angles = (faces_normals * view_direction).sum(dim=1)
    assert cos_angles.mean() < 0, f"The view direction is not correct. cos_angles.mean()={cos_angles.mean()}"
    selected_faces = unique_faces[cos_angles < -eps]

    # find verts
    faces = meshes.faces_packed()[selected_faces]   # [N, 3]
    verts = torch.unique(faces.flatten())   # [N, 1]
    verts_coordinates = meshes.verts_packed()[verts]   # [N, 3]

    # compute color
    if not cameras.is_perspective():
        pt_tensor = cameras.transform_points_ndc(verts_coordinates)[..., :2].squeeze(0) # NDC space points
    else:
        pt_tensor = cameras.transform_points_ndc(verts_coordinates)[..., :2].squeeze(0) # NDC space points
    # valid = ~((pt_tensor.isnan()).any(dim=1))  # checked, correct
    valid = ~((pt_tensor.isnan()|(pt_tensor<-1)|(1<pt_tensor)).any(dim=1))  # checked, correct
    # print("valid",valid.shape)
    valid_pt = pt_tensor[valid, :]
    valid_idx = verts[valid]
    valid_color = torch.nn.functional.grid_sample(image[None], valid_pt[None, :, None, :], align_corners=False, padding_mode="reflection", mode="bilinear")[0, :, :, 0].T.clamp(0, 1)   # [N, 4], note that bicubic may give invalid value
    # valid_color = torch.nn.functional.grid_sample(image[None].flip((-1, -2)), valid_pt[None, :, None, :], align_corners=False, padding_mode="reflection", mode="bilinear")[0, :, :, 0].T.clamp(0, 1)   # [N, 4], note that bicubic may give invalid value
    alpha, valid_color = valid_color[:, 3:], valid_color[:, :3]
    if not use_alpha:
        alpha = torch.ones_like(alpha)

    # modify color
    old_colors = meshes.textures.verts_features_packed()
    old_colors[valid_idx] = valid_color * alpha + old_colors[valid_idx] * (1 - alpha)
    new_texture = TexturesVertex(verts_features=[old_colors])
    
    valid_verts_normals = meshes.verts_normals_packed()[valid_idx]
    valid_verts_normals = valid_verts_normals / valid_verts_normals.norm(dim=1, keepdim=True).clamp_min(0.001)
    cos_angles = (valid_verts_normals * view_direction).sum(dim=1)
    return {
        "new_texture": new_texture,
        "valid_verts": valid_idx,
        "valid_colors": valid_color,
        "valid_alpha": alpha,
        "cos_angles": cos_angles,
    }

def complete_unseen_vertex_color(meshes: Meshes, valid_index: torch.Tensor) -> dict:
    """
    meshes: the mesh with vertex color to be completed.
    valid_index: the index of the valid vertices, where valid means colors are fixed. [V, 1]
    """
    valid_index = valid_index.to(meshes.device)
    colors = meshes.textures.verts_features_packed()    # [V, 3]
    V = colors.shape[0]
    
    invalid_index = torch.ones_like(colors[:, 0]).bool()    # [V]
    invalid_index[valid_index] = False
    invalid_index = torch.arange(V).to(meshes.device)[invalid_index]
    
    L = meshes.laplacian_packed()
    E = torch.sparse_coo_tensor(torch.tensor([list(range(V))] * 2), torch.ones((V,)), size=(V, V)).to(meshes.device)
    L = L + E
    # E = torch.eye(V, layout=torch.sparse_coo, device=meshes.device)
    # L = L + E
    colored_count = torch.ones_like(colors[:, 0])   # [V]
    colored_count[invalid_index] = 0
    L_invalid = torch.index_select(L, 0, invalid_index)    # sparse [IV, V]
    
    total_colored = colored_count.sum()
    coloring_round = 0
    stage = "uncolored"
    from tqdm import tqdm
    pbar = tqdm(miniters=100)
    while stage == "uncolored" or coloring_round > 0:
        new_color = torch.matmul(L_invalid, colors * colored_count[:, None])    # [IV, 3]
        new_count = torch.matmul(L_invalid, colored_count)[:, None]             # [IV, 1]
        colors[invalid_index] = torch.where(new_count > 0, new_color / new_count, colors[invalid_index])
        colored_count[invalid_index] = (new_count[:, 0] > 0).float()
        
        new_total_colored = colored_count.sum()
        if new_total_colored > total_colored:
            total_colored = new_total_colored
            coloring_round += 1
        else:
            stage = "colored"
            coloring_round -= 1
        pbar.update(1)
        if coloring_round > 10000:
            print("coloring_round > 10000, break")
            break
    assert not torch.isnan(colors).any()
    meshes.textures = TexturesVertex(verts_features=[colors])
    return meshes

def multiview_color_projection(meshes: Meshes, image_list: List[Image.Image], cameras_list: List[CamerasBase]=None, camera_focal: float = 2 / 1.35, weights=None, eps=0.05, resolution=1024, device="cuda", reweight_with_cosangle="square", use_alpha=True, confidence_threshold=0.1, complete_unseen=False, below_confidence_strategy="smooth") -> Meshes:
    """
    Projects color from a given image onto a 3D mesh.

    Args:
        meshes (pytorch3d.structures.Meshes): The 3D mesh object, only one mesh.
        image_list (PIL.Image.Image): List of images.
        cameras_list (list): List of cameras.
        camera_focal (float, optional): The focal length of the camera, if cameras_list is not passed. Defaults to 2 / 1.35.
        weights (list, optional): List of weights for each image, for ['front', 'front_right', 'right', 'back', 'left', 'front_left']. Defaults to None.
        eps (float, optional): The threshold for selecting visible faces. Defaults to 0.05.
        resolution (int, optional): The resolution of the projection. Defaults to 1024.
        device (str, optional): The device to use for computation. Defaults to "cuda".
        reweight_with_cosangle (str, optional): Whether to reweight the color with the angle between the view direction and the vertex normal. Defaults to None.
        use_alpha (bool, optional): Whether to use the alpha channel of the image. Defaults to True.
        confidence_threshold (float, optional): The threshold for the confidence of the projected color, if final projection weight is less than this, we will use the original color. Defaults to 0.1.
        complete_unseen (bool, optional): Whether to complete the unseen vertex color using laplacian. Defaults to False.

    Returns:
        Meshes: the colored mesh
    """
    # 1. preprocess inputs
    if image_list is None:
        raise ValueError("image_list is None")
    if cameras_list is None:
        if len(image_list) == 8:
            cameras_list = get_8view_cameras(device, focal=camera_focal)
        elif len(image_list) == 6:
            cameras_list = get_6view_cameras(device, focal=camera_focal)
        elif len(image_list) == 4:
            cameras_list = get_4view_cameras(device, focal=camera_focal)
        elif len(image_list) == 2:
            cameras_list = get_2view_cameras(device, focal=camera_focal)
        else:
            raise ValueError("cameras_list is None, and can not be guessed from image_list")
    if weights is None:
        if len(image_list) == 8:
            weights = [2.0, 0.05, 0.2, 0.02, 1.0, 0.02, 0.2, 0.05]
        elif len(image_list) == 6:
            weights = [2.0, 0.05, 0.2, 1.0, 0.2, 0.05]
        elif len(image_list) == 4:
            weights = [2.0, 0.2, 1.0, 0.2]
        elif len(image_list) == 2:
            weights = [1.0, 1.0]
        else:
            raise ValueError("weights is None, and can not be guessed from image_list")
    
    # 2. run projection
    meshes = meshes.clone().to(device)
    if weights is None:
        weights = [1. for _ in range(len(cameras_list))]
    assert len(cameras_list) == len(image_list) == len(weights)
    original_color = meshes.textures.verts_features_packed()
    assert not torch.isnan(original_color).any()
    texture_counts = torch.zeros_like(original_color[..., :1])
    texture_values = torch.zeros_like(original_color)
    max_texture_counts = torch.zeros_like(original_color[..., :1])
    max_texture_values = torch.zeros_like(original_color)
    for camera, image, weight in zip(cameras_list, image_list, weights):
        ret = project_color(meshes, camera, image, eps=eps, resolution=resolution, device=device, use_alpha=use_alpha)
        if reweight_with_cosangle == "linear":
            weight = (ret['cos_angles'].abs() * weight)[:, None]
        elif reweight_with_cosangle == "square":
            weight = (ret['cos_angles'].abs() ** 2 * weight)[:, None]
        if use_alpha:
            weight = weight * ret['valid_alpha']
        assert weight.min() > -0.0001
        texture_counts[ret['valid_verts']] += weight
        texture_values[ret['valid_verts']] += ret['valid_colors'] * weight
        max_texture_values[ret['valid_verts']] = torch.where(weight > max_texture_counts[ret['valid_verts']], ret['valid_colors'], max_texture_values[ret['valid_verts']])
        max_texture_counts[ret['valid_verts']] = torch.max(max_texture_counts[ret['valid_verts']], weight)

    # Method2
    texture_values = torch.where(texture_counts > confidence_threshold, texture_values / texture_counts, texture_values)
    if below_confidence_strategy == "smooth":
        texture_values = torch.where(texture_counts <= confidence_threshold, (original_color * (confidence_threshold - texture_counts) + texture_values) / confidence_threshold, texture_values)
    elif below_confidence_strategy == "original":
        texture_values = torch.where(texture_counts <= confidence_threshold, original_color, texture_values)
    else:
        raise ValueError(f"below_confidence_strategy={below_confidence_strategy} is not supported")
    assert not torch.isnan(texture_values).any()
    meshes.textures = TexturesVertex(verts_features=[texture_values])
    
    if complete_unseen:
        meshes = complete_unseen_vertex_color(meshes, torch.arange(texture_values.shape[0]).to(device)[texture_counts[:, 0] >= confidence_threshold])
    ret_mesh = meshes.detach()
    del meshes
    return ret_mesh

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def get_sparse_cameras_list(front_view_path, sparse_path, colmap_path, raw_size, input_size, device):
    # get input fxfycxcy
    import os 
    import sys
    front_view_filename = os.path.basename(front_view_path)
    front_view_filename, _ = os.path.splitext(front_view_filename)
    camdata = read_cameras_binary(os.path.join(colmap_path, 'sparse/0/cameras.bin')) 
    imdata = read_images_binary(os.path.join(colmap_path, 'sparse/0/images.bin'))
    H = W = raw_size
    img_wh = (input_size, input_size)
    factor_x = factor_y = 1
    input_size = 2048
    # factor_x = input_size / W
    # factor_y = input_size / H
    print("[INFO] factor",factor_x, input_size, W)
    if camdata[1].model == 'SIMPLE_RADIAL':
        fx = camdata[1].params[0] * factor_x
        fy = camdata[1].params[0] * factor_y / 1000
        cx = camdata[1].params[1] * factor_x / 1000
        cy = camdata[1].params[2] * factor_y / 1000
    elif camdata[1].model in ['PINHOLE', 'OPENCV']:
        fx = camdata[1].params[0] * factor_x / 1000
        fy = camdata[1].params[1] * factor_y / 1000
        cx = camdata[1].params[2] * factor_x / 1000
        cy = camdata[1].params[3] * factor_y / 1000
    else:
        raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
    height = camdata[1].height
    width = camdata[1].width
    if camdata[1].model=="SIMPLE_PINHOLE":
        focal_length_x = camdata[1].params[0]
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)
    elif camdata[1].model=="PINHOLE":
        focal_length_x = camdata[1].params[0]
        focal_length_y = camdata[1].params[1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
    elif camdata[1].model=="OPENCV":
        focal_length_x = camdata[1].params[0]
        focal_length_y = camdata[1].params[1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        
    fxfycxcy = np.array([fx,fy,cx,cy])
   

    fovx = np.degrees(2 * np.arctan(0.5 * input_size / fx))
    fovy = np.degrees(2 * np.arctan(0.5 * input_size / fy))
    print("[INFO] fov:",fovx, fovy)
    import os
    ret = {"sparse": [] }
    c2ws = []
    front_view_camera = None
    ones_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
    front_back_view_camera_truth = get_front_camera(azim_list = [0,180], focal=1.) # Pytorch3D
    front_back_view_camera_truth = [torch.cat((c, ones_row), dim=0).double() for c in front_back_view_camera_truth]
    front_view_camera_truth = front_back_view_camera_truth[0]
    # front_view_camera_truth[[1,2]] = front_view_camera_truth[[2,1]]
    back_view_camera_truth = front_back_view_camera_truth[1]
    # front_view_camera_truth[0] *= -1
    # front_view_camera_truth[2] *= -1
    K = torch.zeros((3, 4))
    # image_resolution_width = 1080  # 图像分辨率宽度，像素
    # image_resolution_height = 1920  # 图像分辨率高度，像素
    # sensor_width_mm = 36  # 传感器宽度，毫米
    # fx = fy = 50 
    # # 计算 PPU
    # ppu = image_resolution_width / sensor_width_mm
    # fx = fx * ppu
    # fy = fy * ppu
    # cx = image_resolution_width / 2
    # cy = image_resolution_height / 2
    K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[2, 2]  = fx, fy, cx, cy, 1
    K = torch.cat((K, ones_row), dim=0).unsqueeze(0).double()
    t = get_front_camera(azim_list = [0,180], focal=1.) # Pytorch3D

    # front_view_camera_truth[0:3,:] *= -1
    print("front_view_camera_truth",t)
    
    
    if front_view_camera is None:
        extrinsic = imdata[int(front_view_filename.lstrip("0"))]
        R = torch.tensor(extrinsic.qvec2rotmat()).double().unsqueeze(0)
        T = torch.tensor(extrinsic.tvec.reshape(1, 3)).double()
        front_view_w2c = torch.cat([R[0], T[0, :, None]], dim=1).double()
        front_view_w2c = torch.cat((front_view_w2c, ones_row), dim=0).double()
        print(front_view_w2c.shape)

        front_view_w2c[:,1:3] *= -1. # COLMAP => Pytorch3D

        # front_view_w2c[[1,2]] = front_view_w2c[[2,1]]

        print("front_view", front_view_w2c)
        # front_view tensor([[-0.9997, -0.0133,  0.0217,  0.2169],
        # [-0.0211, -0.0382, -0.9990, -2.6005],
        # [ 0.0141, -0.9992,  0.0379,  1.5357],
        # [ 0.0000,  0.0000,  0.0000,  1.0000]])
        # front_view_w2c = front_view_w2c.transpose(-2,-1)
        front_view_camera: PerspectiveCameras = get_camera(world_to_cam=front_view_w2c,
                                                            # focal_length=1.0, cam_type='orthogonal').to(device)
                                                                 cam_type='fov').to(device)
        ret["front"] = front_view_camera
    # transform_matrix = torch.matmul(front_view_camera_truth, front_view_w2c.inverse())
    # print(transform_matrix)
    L = -5
    move_transform = torch.eye(4, dtype=torch.float64)
    move_transform[2, 3] = L
    move_transform_center = torch.eye(4, dtype=torch.float64)
    move_transform_center[2, 3] = L / 2
    rotate_transform = torch.tensor([[ -1,  0,  0, 0],
                                    [  0,  1,  0, 0],
                                    [  0,  0, -1, 0],
                                    [  0,  0,  0, 1]], dtype=torch.float64)
    
    center_point_w2c = move_transform_center @ front_view_w2c
    print("center_point_w2c",center_point_w2c)
    back_view_w2c = rotate_transform @ move_transform @ front_view_w2c
    # back_view_w2c = back_view_w2c.transpose(-2,-1)
    back_view_camera: PerspectiveCameras = get_camera(world_to_cam=back_view_w2c,
                                                            # focal_length=1.0, cam_type='orthogonal').to(device)
                                                                 cam_type='fov').to(device)
    ret["back"] = back_view_camera
    for filename in os.listdir(sparse_path):   
        if filename.endswith(".jpg") or filename.endswith(".png"):
            f, _ = os.path.splitext(filename)
            # get input pose
            extrinsic = imdata[int(f.lstrip("0"))]
            R = torch.tensor(extrinsic.qvec2rotmat()).double().unsqueeze(0)
            T = torch.tensor(extrinsic.tvec.reshape(1, 3)).double()
            
            w2c = torch.cat([R[0], T[0, :, None]], dim=1)
            w2c = torch.cat((w2c, ones_row), dim=0)

            w2c[:,1:3] *= -1. # COLMAP => Pytorch3D

            # w2c[[1,2]] = w2c[[2,1]]
            # w2c[0:3,:] *= -1. # COLMAP => Pytorch3D
            print("other_view",w2c)

            # w2c = torch.matmul(transform_matrix, w2c)
            
            # print("other_view_truth",w2c)

            # w2c = front_view_camera_truth

            # w2c[[1,2]] = w2c[[2,1]]

            # w2c[0:3,:] *= -1

            # w2c[:3,:3] = w2c[:3,:3].T

            # w2c = w2c.transpose(-2,-1)

            # w2c = w2c[:3, :]

            # cameras: OrthographicCameras = get_camera(world_to_cam=w2c, focal_length=1.0, cam_type='ortho')
            #                                         # cam_type='fov')
            
            cameras: PerspectiveCameras =  get_camera(world_to_cam=w2c, cam_type='fov').to(device)
                                                    # cam_type='fov')
            ret["sparse"].append(cameras)
    
    
    return ret


def compute_c2w_matrix(x, y, z, rot_x_deg, rot_y_deg, rot_z_deg):
    # 将角度转换为弧度
    rot_x_rad = math.radians(rot_x_deg)
    rot_y_rad = math.radians(rot_y_deg)
    rot_z_rad = math.radians(rot_z_deg)

    # 计算绕x轴的旋转矩阵
    Rx = torch.tensor([
        [1, 0, 0, 0],
        [0, math.cos(rot_x_rad), -math.sin(rot_x_rad), 0],
        [0, math.sin(rot_x_rad), math.cos(rot_x_rad), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # 计算绕y轴的旋转矩阵
    Ry = torch.tensor([
        [math.cos(rot_y_rad), 0, math.sin(rot_y_rad), 0],
        [0, 1, 0, 0],
        [-math.sin(rot_y_rad), 0, math.cos(rot_y_rad), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # 计算绕z轴的旋转矩阵
    Rz = torch.tensor([
        [math.cos(rot_z_rad), -math.sin(rot_z_rad), 0, 0],
        [math.sin(rot_z_rad), math.cos(rot_z_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # 计算总的旋转矩阵
    R = torch.matmul(torch.matmul(Rz, Ry), Rx)
    R = R.t()
    # rotate_z_inverse = torch.tensor([[1,  0,  0, 0],
    #                              [0,  1,  0, 0],
    #                              [0,  0, -1, 0],
    #                              [0,  0,  0, 1]], dtype=torch.float32)
    
    # R = torch.matmul(rotate_z_inverse, R)

    # 创建平移向量，并应用旋转
    T = torch.tensor([x, y, z, 1], dtype=torch.float32)
    # T = -torch.matmul(R, T)[:3]

    # 将旋转矩阵和平移向量组合成仿射变换矩阵
    # c2w = torch.eye(4, dtype=torch.float32)
    c2w = R.clone()
    c2w[:, 3] = T

    return c2w


def get_sparse_cameras_list_orth(front_view_path, sparse_path, raw_size, input_size, device):
    import os 
    import sys
    front_view_filename = os.path.basename(front_view_path)
    front_view_filename, _ = os.path.splitext(front_view_filename)
    H = W = raw_size
    img_wh = (input_size, input_size)
    factor_x = factor_y = 1
    input_size = 2048
 
    ret = {"sparse": [] }
    c2ws = []
    front_view_camera = None
    ones_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)

    front_view_w2c = compute_c2w_matrix(0.01, -0.05, 3.56, 0, 180, 0).double()
    # front_view_w2c = torch.cat((front_view_w2c, ones_row), dim=0).double()
    print(front_view_w2c.shape,front_view_w2c)

    # front_view_w2c[:,1:3] *= -1. # COLMAP => Pytorch3D

    # front_view_w2c[[1,2]] = front_view_w2c[[2,1]]

    front_view_camera: OrthographicCameras = get_camera(world_to_cam=front_view_w2c,
                                                        focal_length=1.0, cam_type='orthogonal').to(device)
                                                                # cam_type='fov').to(device)
    ret["front"] = front_view_camera
    # transform_matrix = torch.matmul(front_view_camera_truth, front_view_w2c.inverse())
    # print(transform_matrix)
    L = -10
    move_transform = torch.eye(4, dtype=torch.float64)
    move_transform[2, 3] = L
    move_transform_center = torch.eye(4, dtype=torch.float64)
    move_transform_center[2, 3] = L / 2
    rotate_transform = torch.tensor([[ -1,  0,  0, 0],
                                    [  0,  1,  0, 0],
                                    [  0,  0, -1, 0],
                                    [  0,  0,  0, 1]], dtype=torch.float64)
    
    center_point_w2c = move_transform_center @ front_view_w2c
    print("center_point_w2c",center_point_w2c)
    back_view_w2c = rotate_transform @ move_transform @ front_view_w2c
    # back_view_w2c = back_view_w2c.transpose(-2,-1)
    back_view_camera: OrthographicCameras = get_camera(world_to_cam=back_view_w2c,
                                                            focal_length=1.0, cam_type='orthogonal').to(device)
                                                                #  cam_type='fov').to(device)
    ret["back"] = back_view_camera




    other_c2w_2 = compute_c2w_matrix(1.85, -0.05, 1.72, 0, 225, 0).double()
    # other_c2w_2[:,1:3] *= -1. # COLMAP => Pytorch3D
    # other_c2w_2[[1,2]] = other_c2w_2[[2,1]]
    other_view_camera: OrthographicCameras = get_camera(world_to_cam=other_c2w_2,
                                                focal_length=1.0, cam_type='orthogonal').to(device)
    ret["sparse"].append(other_view_camera)


    other_c2w_1 = compute_c2w_matrix(-2.096, -0.067, 2.034, 0, 135, 0).double()
    # other_c2w_1[:,1:3] *= -1. # COLMAP => Pytorch3D
    # other_c2w_1[[1,2]] = other_c2w_1[[2,1]]
    other_view_camera: OrthographicCameras = get_camera(world_to_cam=other_c2w_1,
                                                focal_length=1.0, cam_type='orthogonal').to(device)
    ret["sparse"].append(other_view_camera)


    ret["sparse"].append(front_view_camera)
    return ret



def get_front_camera(azim_list, focal=2/1.35, dist=1.1):
    ret = []
    for azim in azim_list:
        R, T = look_at_view_transform(dist, 0, azim)
        w2c = torch.cat([R[0].T, T[0, :, None]], dim=1).double()
        ret.append(w2c)
    return ret

def get_cameras_list(azim_list, device, focal=2/1.35, dist=1.1):
    ret = []
    for azim in azim_list:
        R, T = look_at_view_transform(dist, 0, azim)
        w2c = torch.cat([R[0].T, T[0, :, None]], dim=1)
        cameras: OrthographicCameras = get_camera(w2c, focal_length=focal, cam_type='orthogonal').to(device)
        ret.append(cameras)
    return ret

def get_8view_cameras(device, focal=2/1.35):
    return get_cameras_list(azim_list = [180, 225, 270, 315, 0, 45, 90, 135], device=device, focal=focal)

def get_6view_cameras(device, focal=2/1.35):
    return get_cameras_list(azim_list = [180, 225, 270, 0, 90, 135], device=device, focal=focal)

def get_4view_cameras(device, focal=2/1.35):
    return get_cameras_list(azim_list = [180, 270, 0, 90], device=device, focal=focal)

def get_2view_cameras(device, focal=2/1.35):
    return get_cameras_list(azim_list = [180, 0], device=device, focal=focal)

def get_multiple_view_cameras(device, focal=2/1.35, offset=180, num_views=8, dist=1.1):
    return get_cameras_list(azim_list = (np.linspace(0, 360, num_views+1)[:-1] + offset) % 360, device=device, focal=focal, dist=dist)

def align_with_alpha_bbox(source_img, target_img, final_size=1024):
    # align source_img with target_img using alpha channel
    # source_img and target_img are PIL.Image.Image
    source_img = source_img.convert("RGBA")
    target_img = target_img.convert("RGBA").resize((final_size, final_size))
    source_np = np.array(source_img)
    target_np = np.array(target_img)
    source_alpha = source_np[:, :, 3]
    target_alpha = target_np[:, :, 3]
    bbox_source_min, bbox_source_max = np.argwhere(source_alpha > 0).min(axis=0), np.argwhere(source_alpha > 0).max(axis=0)
    bbox_target_min, bbox_target_max = np.argwhere(target_alpha > 0).min(axis=0), np.argwhere(target_alpha > 0).max(axis=0)
    source_content = source_np[bbox_source_min[0]:bbox_source_max[0]+1, bbox_source_min[1]:bbox_source_max[1]+1, :]
    # resize source_content to fit in the position of target_content
    source_content = Image.fromarray(source_content).resize((bbox_target_max[1]-bbox_target_min[1]+1, bbox_target_max[0]-bbox_target_min[0]+1), resample=Image.BICUBIC)
    target_np[bbox_target_min[0]:bbox_target_max[0]+1, bbox_target_min[1]:bbox_target_max[1]+1, :] = np.array(source_content)
    return Image.fromarray(target_np)
    
def load_image_list_from_mvdiffusion(mvdiffusion_path, front_from_pil_or_path=None):
    import os
    image_list = []
    for dir in ['front', 'front_right', 'right', 'back', 'left', 'front_left']:
        image_path = os.path.join(mvdiffusion_path, f"rgb_000_{dir}.png")
        pil = Image.open(image_path)
        if dir == 'front':
            if front_from_pil_or_path is not None:
                if isinstance(front_from_pil_or_path, str):
                    replace_pil = Image.open(front_from_pil_or_path)
                else:
                    replace_pil = front_from_pil_or_path
                # align replace_pil with pil using bounding box in alpha channel
                pil = align_with_alpha_bbox(replace_pil, pil, final_size=1024)
        image_list.append(pil)
    return image_list

def load_image_list_from_img_grid(img_grid_path, resolution = 1024):
    img_list = []
    grid = Image.open(img_grid_path)
    w, h = grid.size
    for row in range(0, h, resolution):
        for col in range(0, w, resolution):
            img_list.append(grid.crop((col, row, col + resolution, row + resolution)))
    return img_list