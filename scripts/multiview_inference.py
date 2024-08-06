import os
from PIL import Image
from scripts.mesh_init import build_mesh, calc_w_over_h, fix_border_with_pymeshlab_fast
from scripts.project_mesh import multiview_color_projection
from scripts.refine_lr_to_sr import run_sr_fast
from scripts.utils import simple_clean_mesh
from app.utils import simple_remove, split_image, extract_obj_and_resize_images, resize_and_place_images, extract_obj
from app.custom_models.normal_prediction import predict_normals
from mesh_reconstruction.recon import reconstruct_stage1
from mesh_reconstruction.refine import run_mesh_refine
from scripts.project_mesh import get_cameras_list
from scripts.utils import from_py3d_mesh, to_pyml_mesh
from pytorch3d.structures import Meshes, join_meshes_as_scene
import numpy as np
import torch

def fast_geo(front_normal: Image.Image, back_normal: Image.Image, side_normal: Image.Image, front_camera, clamp=0., init_type="std"):
    import time
    if front_normal.mode == "RGB":
        front_normal = simple_remove(front_normal, run_sr=False)
    front_normal = front_normal.resize((192, 192))
    if back_normal.mode == "RGB":
        back_normal = simple_remove(back_normal, run_sr=False)
    back_normal = back_normal.resize((192, 192))
    if side_normal.mode == "RGB":
        side_normal = simple_remove(side_normal, run_sr=False)
    side_normal = side_normal.resize((192, 192))
    
    # build mesh with front back projection # ~3s
    side_w_over_h = calc_w_over_h(back_normal) #TODO: back or side
    mesh_front = build_mesh(front_normal, front_normal, clamp_min=clamp, scale=side_w_over_h, init_type=init_type)
    mesh_back = build_mesh(back_normal, back_normal, is_back=True, clamp_min=clamp, scale=side_w_over_h, init_type=init_type)
    meshes = join_meshes_as_scene([mesh_front, mesh_back])
    meshes = fix_border_with_pymeshlab_fast(meshes, poissson_depth=6, simplification=2000)
    T = front_camera.T
    device = meshes.verts_packed().device
    # translation_vector = torch.tensor([0, 0, 0], dtype=torch.float32, device=device).to(device) #0.2169  -1.5357
    
    translation_vector = torch.tensor([1.0000e-02, -5.0000e-02, -1.4400e+00], dtype=torch.float32, device=device).to(device)
    # translation_vector = torch.tensor([0.2169, -1.5357, 0.1005], dtype=torch.float32, device=device).to(device)
    print("translation_vector",translation_vector)
    # translation_vector[:,[1,2]] = translation_vector[:,[2,1]]
    # translation_vector[1:3] *= -1
    # scale_factor = 1.5
    translated_verts_list = []
    for verts in meshes.verts_list():
        # verts = verts * scale_factor
        translated_verts = verts + translation_vector[:]
        translated_verts_list.append(translated_verts)

    meshes._verts_list = translated_verts_list


    from pytorch3d.io import save_obj
    verts = meshes.verts_list()[0]
    faces = meshes.faces_list()[0]
    save_obj("croase_offset.obj", verts, faces)
    return meshes

def refine_rgb(rgb_pils, front_pil):
    from scripts.refine_lr_to_sr import refine_lr_with_sd
    from scripts.utils import NEG_PROMPT
    from app.utils import make_image_grid
    from app.all_models import model_zoo
    from app.utils import rgba_to_rgb
    rgb_pil = make_image_grid(rgb_pils, rows=2)
    prompt = "4views, multiview"
    neg_prompt = NEG_PROMPT
    control_image = rgb_pil.resize((1024, 1024))
    refined_rgb = refine_lr_with_sd([rgb_pil], [rgba_to_rgb(front_pil)], [control_image], prompt_list=[prompt], neg_prompt_list=[neg_prompt], pipe=model_zoo.pipe_disney_controlnet_tile_ipadapter_i2i, strength=0.2, output_size=(1024, 1024))[0]
    refined_rgbs = split_image(refined_rgb, rows=2)
    return refined_rgbs

def erode_alpha(img_list):
    out_img_list = []
    for idx, img in enumerate(img_list):
        arr = np.array(img)
        alpha = (arr[:, :, 3] > 127).astype(np.uint8)
        # erode 1px
        import cv2
        alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=1)
        alpha = (alpha * 255).astype(np.uint8)
        img = Image.fromarray(np.concatenate([arr[:, :, :3], alpha[:, :, None]], axis=-1))
        out_img_list.append(img)
    return out_img_list
import time


def geo_reconstruct(rgb_pils, normal_pils, sparse_pils, sparse_cameras, front_pil, do_refine=False, predict_normal=True, expansion_weight=0.1, init_type="std"):
    print("[INFO] rgb_pils: ", len(rgb_pils))
    print("[INFO] sparse_pils: ", len(sparse_pils))
    print("[INFO] sparse_pils size: ", sparse_pils[0].size[0], sparse_pils[0].size[1])

    # if front_pil.size[0] <= 512:
    #     front_pil = run_sr_fast([front_pil])[0]  #TODO
    # if sparse_pils[0].size[0] <= 512:
    #     print("[INFO] run_sr_fast sparse_pils")
    #     sparse_pils = run_sr_fast(sparse_pils) #TODO
    if do_refine:
        refined_rgbs = refine_rgb(rgb_pils, front_pil)  # 6s
    else:
        refined_rgbs = [rgb.resize((512, 512), resample=Image.LANCZOS) for rgb in rgb_pils]
    # img_list = [front_pil] + run_sr_fast(refined_rgbs[1:]) #TODO
    img_list = [front_pil] + refined_rgbs[1:]
    # img_list = [front_pil] + refined_rgbs[1:]
    sparse_list = sparse_pils
    front_camera_list = get_cameras_list([0],"cuda", focal=1)
    
    if predict_normal:
        rm_normals = predict_normals([img.resize((512, 512), resample=Image.LANCZOS) for img in img_list], guidance_scale=1.5)
        normal_sparse_cameras = [sparse_cameras["front"]] + sparse_cameras["sparse"]
        # normal_sparse_cameras = front_camera_list + sparse_cameras
        for i, img in enumerate(sparse_list):
            r, g, b, a = img.split()
            a.save(f'sparse_list_alpha{i}.jpg')
            # img.convert('RGB').save(f'sparse_list{i}.jpg')
        # sparse_list, box_sizes, min_max_coords_list = extract_obj_and_resize_images(sparse_list, target_size=(512,512))
        # print(box_sizes)
        # print(min_max_coords_list)
        
        sparse_normals = predict_normals([img for img in sparse_list],
                                         sparse=True, sparse_cameras=normal_sparse_cameras, guidance_scale=1.5)
        # sparse_normals = resize_and_place_images(sparse_normals, target_size=(512,512), box_sizes=box_sizes, min_max_coords_list=min_max_coords_list)
        # sparse_normals[0], sparse_normals[1] = sparse_normals[1], sparse_normals[0]
    else:
        rm_normals = simple_remove([img.resize((512, 512), resample=Image.LANCZOS) for img in normal_pils])
    # transfer the alpha channel of rm_normals to img_list
    for idx, img in enumerate(rm_normals):
        if idx == 0 and img_list[0].mode == "RGBA":
            temp = img_list[0].resize((512, 512)) #TODO
            rm_normals[0] = Image.fromarray(np.concatenate([np.array(rm_normals[0])[:, :, :3], np.array(temp)[:, :, 3:4]], axis=-1))
            continue
        img_list[idx] = Image.fromarray(np.concatenate([np.array(img_list[idx]), np.array(img)[:, :, 3:4]], axis=-1))
    # for idx, img in enumerate(sparse_normals):
    #     temp = sparse_list[idx].resize((512, 512)) #TODO
    #     sparse_normals[idx] = Image.fromarray(np.concatenate([np.array(sparse_normals[idx])[:, :, :3], np.array(temp)[:, :, 3:4]], axis=-1))
    #     sparse_list[idx] = Image.fromarray(np.concatenate([np.array(sparse_list[idx])[..., :3], np.array(img)[:, :, 3:4]], axis=-1)) #apply mask

    assert img_list[0].mode == "RGBA"
    assert sparse_list[0].mode == "RGBA"
    assert np.mean(np.array(img_list[0])[..., 3]) < 250
    assert np.mean(np.array(sparse_list[0])[..., 3]) < 250

    # img_list = erode_alpha(img_list[1:])
    img_list = [img_list[0]] + erode_alpha(img_list[1:])
    normal_stg1 = [img.resize((512, 512)) for img in rm_normals]

    sparse_normal_stg1 = [img.resize((512, 512)) for img in sparse_normals]
    normal_stg1[0].convert('RGB').save('normal_0.jpg')
    normal_stg1[2].convert('RGB').save('normal_2.jpg')
    front_normal = [normal_stg1[0]]
    back_normal = [normal_stg1[2]]
    
    print("[INFO] init type: ", init_type)
    # normal_stg1 = normal_stg1[1:]
    box_sizes, min_max_coords_list = extract_obj([sparse_pils[-1]])
    print(box_sizes)
    print(min_max_coords_list)
    back_normal_resize, _, _ = extract_obj_and_resize_images(back_normal, target_size=box_sizes[0])
    back_normal = resize_and_place_images(back_normal_resize, target_size=(512,512), box_sizes=box_sizes, min_max_coords_list=min_max_coords_list)
    t = back_normal[0].convert("RGB")
    t.save("back_percep.jpg")
    normal_stg1 = back_normal + sparse_normal_stg1
    
    if init_type in ["std", "thin"]:
        meshes = fast_geo(sparse_normals[-1], back_normal[0], normal_stg1[2], front_camera=sparse_cameras["front"], init_type=init_type) #side view
        # _ = multiview_color_projection(meshes, rgb_pils, resolution=512, device="cuda", complete_unseen=False, confidence_threshold=0.1)    # just check for validation, may throw error
        vertices, faces, _ = from_py3d_mesh(meshes)
        vertices, faces = vertices.to("cuda"), faces.to("cuda")
        sparse_cameras["front"], sparse_cameras["back"] = sparse_cameras["front"].to(device=vertices.device), sparse_cameras["back"].to(device=vertices.device)
        sparse_cameras["sparse"] = [c.to(device=vertices.device) for c in sparse_cameras["sparse"]]
        
        vertices, faces = reconstruct_stage1(normal_stg1, steps=200, vertices=vertices, faces=faces,
                                             sparse_cameras=[sparse_cameras["back"]] + sparse_cameras["sparse"],
                                              start_edge_len=0.1, end_edge_len=0.02, gain=0.05, return_mesh=False, loss_expansion_weight=expansion_weight)
    elif init_type in ["ball"]:
        meshes = fast_geo(sparse_normals[2], back_normal[0], normal_stg1[2], sparse_cameras=sparse_cameras["front"], init_type=init_type) #side view
        # _ = multiview_color_projection(meshes, rgb_pils, resolution=512, device="cuda", complete_unseen=False, confidence_threshold=0.1)    # just check for validation, may throw error
        vertices, faces, _ = from_py3d_mesh(meshes)
        vertices, faces = vertices.to("cuda"), faces.to("cuda")
        sparse_cameras["front"], sparse_cameras["back"]= sparse_cameras["front"].to(device=vertices.device), sparse_cameras["back"].to(device=vertices.device)
        sparse_cameras["sparse"] = [c.to(device=vertices.device) for c in sparse_cameras["sparse"]]

        vertices, faces = reconstruct_stage1(normal_stg1, steps=200, vertices=vertices, faces=faces,
                                              sparse_cameras=[sparse_cameras["back"]] + sparse_cameras["sparse"],
                                              end_edge_len=0.01, return_mesh=False, loss_expansion_weight=expansion_weight)
    
    # TODO: back
    img_list = [img_list[2]]
    img_list = img_list + sparse_list
    # TODO: front back
    t = rm_normals[2].convert("RGB")
    t.save("diffusion_2.jpg")
    t = rm_normals[3].convert("RGB")
    t.save("diffusion_3.jpg")

    # back_normal = [rm_normals[2]]
    rm_normals = back_normal + sparse_normals
    
    vertices, faces = run_mesh_refine(vertices, faces, rm_normals, sparse_cameras=[sparse_cameras["back"]] + sparse_cameras["sparse"], 
                                      steps=150, start_edge_len=0.02, end_edge_len=0.005, decay=0.99, update_normal_interval=20, update_warmup=5, return_mesh=False, process_inputs=False, process_outputs=False)
    meshes = simple_clean_mesh(to_pyml_mesh(vertices, faces), apply_smooth=True, stepsmoothnum=1, apply_sub_divide=True, sub_divide_threshold=0.25).to("cuda")
    print(len(img_list), len([1.0]  + [2.0] * (len(sparse_cameras["sparse"])-1)),
            len([sparse_cameras["back"]] + sparse_cameras["sparse"]))
    new_meshes = multiview_color_projection(meshes, img_list, resolution=1024, device="cuda", 
                                            weights=[1.0]+[1.0]*len(sparse_cameras["sparse"]),
                                            # weights=[2.0, 0.8, 1.0, 0.8] + [2.0] * len(sparse_cameras), 
                                            complete_unseen=True, confidence_threshold=0.2, 
                                            # cameras_list = (get_cameras_list([0, 90, 180, 270], "cuda", focal=1) + sparse_cameras)
                                            cameras_list=[sparse_cameras["back"]]+sparse_cameras["sparse"]
                                            )
    return new_meshes
