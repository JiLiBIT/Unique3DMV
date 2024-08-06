# modified from https://github.com/Profactor/continuous-remeshing
import nvdiffrast.torch as dr
import torch
from typing import Tuple

def _warmup(glctx, device=None):
    device = 'cuda' if device is None else device
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device=device, **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

glctx = dr.RasterizeGLContext(output_db=False, device="cuda")




class NormalsRenderer:
    
    _glctx:dr.RasterizeGLContext = None
    
    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: Tuple[int,int],
            sparse_cameras = None,
            mvp = None,
            device=None,
            ):
        if mvp is None:
            if mv is not None and proj is not None:
                self._mvp = proj @ mv #C,4,4
            else:
                self._mvp = None
        else:
            self._mvp = mvp
        self.sparse_RT = None
        self.sparse_intrinsics = None
        if sparse_cameras is not None:
            sparse_mvp = []
            sparse_RT = []
            sparse_intrinsics = []
            
            from mesh_reconstruction.func import _orthographic, _projection, getProjectionMatrix
            for camera in sparse_cameras:
                # world_to_view_transform = camera.get_world_to_view_transform(R=camera.R, T=camera.T).get_matrix()
                # world_to_view_transform = world_to_view_transform.T.reshape(-1,4,4)
                # model_view_projection_matrix = proj1 @ world_to_view_transform
                # print("render_view", world_to_view_transform)
                world_to_view_transform = camera.get_world_to_view_transform(R=camera.R, T=camera.T).get_matrix().float()
                # world_to_view_transform = world_to_view_transform.T.reshape(-1,4,4)
                print("world_to_view_transform1111", world_to_view_transform.shape, world_to_view_transform)
                proj1 = camera.get_projection_transform().get_matrix().float()
                intrinsic_matrix = proj1
                model_view_projection_matrix1 = proj1 @ world_to_view_transform

                model_view_projection_matrix = camera.get_full_projection_transform().get_matrix().float()
                
                # print("model_view_projection_matrix111",model_view_projection_matrix)
                # print("model_view_projection_matrix",model_view_projection_matrix1)
                # print("model_view_projection_matrix2222",model_view_projection_matrix1)
                sparse_mvp.append(model_view_projection_matrix.float())
                sparse_RT.append(world_to_view_transform)
                sparse_intrinsics.append(intrinsic_matrix)
            sparse_mvp = torch.stack(sparse_mvp, dim=0).squeeze(1) #C,4,4
            self.sparse_RT = torch.stack(sparse_RT, dim=0).squeeze(1) #C,4,4
            self.sparse_intrinsics = torch.stack(sparse_intrinsics, dim=0).squeeze(1) #C,4,4
            # sparse_cameras = torch.stack(sparse_cameras, dim=0) 
            if self._mvp is not None:
                self._mvp = torch.cat((self._mvp, sparse_mvp), dim=0)
            else:
                self._mvp = sparse_mvp
            # plot_pose(self._mvp, './', 'pose')
            
        self._image_size = image_size
        self._glctx = glctx
        _warmup(self._glctx, device)

    def render(self,
            vertices: torch.Tensor, #V,3 float
            normals: torch.Tensor, #V,3 float   in [-1, 1]
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32) 
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        # vertices_clip = vert_hom @ self._mvp #C,V,4
        vertices_clip = vert_hom @ self._mvp #.transpose(-2,-1) #C,V,4
        # vertices_clip = vertices_clip[:3] / vertices_clip[-1] 
        # vertices_clip = torch.cat((vertices_clip, torch.ones(vertices_clip.shape[0],vertices_clip.shape[1],1,device=vertices.device)),axis=-1) #V,3 -> V,4
        # print("vertices_clip",vertices_clip.shape)#C,V,4 torch.Size([6, 1827, 4])apt 
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        import imageio
        rast_out_image = (rast_out * 255).to(torch.uint8).cpu()
        image_data = rast_out_image[0]
        imageio.imwrite('render_rast.png', image_data)

        vert_col = (normals+1)/2 #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4



from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,
    TexturesVertex,
    MeshRasterizer,
    BlendParams,
    FoVOrthographicCameras,
    look_at_view_transform,
    hard_rgb_blend,
)

class VertexColorShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        return hard_rgb_blend(texels, fragments, blend_params)

def render_mesh_vertex_color(mesh, cameras, H, W, blur_radius=0.0, faces_per_pixel=1, bkgd=(0., 0., 0.), dtype=torch.float32, device="cuda"):
    if len(mesh) != len(cameras):
        if len(cameras) % len(mesh) == 0:
            mesh = mesh.extend(len(cameras))
        else:
            raise NotImplementedError()
    
    # render requires everything in float16 or float32
    input_dtype = dtype
    blend_params = BlendParams(1e-4, 1e-4, bkgd)

    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        clip_barycentric_coords=True,
        bin_size=None,
        max_faces_per_bin=None,
    )

    # Create a renderer by composing a rasterizer and a shader
    # We simply render vertex colors through the custom VertexColorShader (no lighting, materials are used)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=VertexColorShader(
            device=device,
            cameras=cameras,
            blend_params=blend_params
        )
    )

    # render RGB and depth, get mask
    with torch.autocast(dtype=input_dtype, device_type=torch.device(device).type):
        images, _ = renderer(mesh)
    return images   # BHW4

class Pytorch3DNormalsRenderer: # 100 times slower!!!
    def __init__(self, cameras, image_size, device):
        self.cameras = cameras.to(device)
        self._image_size = image_size
        self.device = device
    
    def render(self,
            vertices: torch.Tensor, #V,3 float
            normals: torch.Tensor, #V,3 float   in [-1, 1]
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,4
        mesh = Meshes(verts=[vertices], faces=[faces], textures=TexturesVertex(verts_features=[(normals + 1) / 2])).to(self.device)
        return render_mesh_vertex_color(mesh, self.cameras, self._image_size[0], self._image_size[1], device=self.device)
    
def save_tensor_to_img(tensor, save_dir):
    from PIL import Image
    import numpy as np
    for idx, img in enumerate(tensor):
        img = img[..., :3].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(save_dir + f"{idx}.png")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mesh_reconstruction.func import make_star_cameras_orthographic, make_star_cameras_orthographic_py3d
    cameras = make_star_cameras_orthographic_py3d([0, 270, 180, 90], device="cuda", focal=1., dist=4.0)
    mv,proj = make_star_cameras_orthographic(4, 1)
    resolution = 1024
    renderer1 = NormalsRenderer(mv,proj, [resolution,resolution], device="cuda")
    renderer2 = Pytorch3DNormalsRenderer(cameras, [resolution,resolution], device="cuda")
    vertices = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[1,0,0]], device="cuda", dtype=torch.float32)
    normals = torch.tensor([[-1,-1,-1],[1,-1,-1],[-1,-1,1],[-1,1,-1]], device="cuda", dtype=torch.float32)
    faces = torch.tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], device="cuda", dtype=torch.long)
    
    import time
    t0 = time.time()
    r1 = renderer1.render(vertices, normals, faces)
    print("time r1:", time.time() - t0)
    
    t0 = time.time()
    r2 = renderer2.render(vertices, normals, faces)
    print("time r2:", time.time() - t0)
    
    for i in range(4):
        print((r1[i]-r2[i]).abs().mean(), (r1[i]+r2[i]).abs().mean())