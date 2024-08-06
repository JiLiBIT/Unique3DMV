import os
import gradio as gr
from PIL import Image
from pytorch3d.structures import Meshes
from app.utils import clean_up
from app.custom_models.mvimg_prediction import run_mvprediction
from app.custom_models.normal_prediction import predict_normals
from scripts.refine_lr_to_sr import run_sr_fast
from scripts.utils import save_glb_and_video
from scripts.multiview_inference import geo_reconstruct
from scripts.project_mesh import multiview_color_projection, get_cameras_list, get_sparse_cameras_list, get_sparse_cameras_list_orth

def generate3dv2(preview_path, input_processing, seed, render_video=True, do_refine=True, expansion_weight=0.1, init_type="ball"):
    # preview_path = '../data/snow_boy/o_i/0069.jpg'
    # sparse_path = '../data/snow_boy/images/'
    preview_path = '../data/snow_boy_orth/2.png'
    sparse_path = '../data/snow_boy_orth/'

    if preview_path is None:
        raise gr.Error("preview_img is none")
    if isinstance(preview_path, str):
        preview_img = Image.open(preview_path)
    sparse_pils = []
    
    
    from rembg import remove
    from scripts.utils import session, simple_preprocess, expand2square
    from app.utils import change_rgba_bg, rgba_to_rgb
    for filename in os.listdir(sparse_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print("[INFO] filename:",filename)
            if os.path.join(preview_path) == os.path.join(sparse_path, filename):
                continue
            img = Image.open(os.path.join(sparse_path, filename))
            img = remove(img, session=session)
            img = change_rgba_bg(img, "white")
            img.convert('RGB').save("rgb_pils_0.jpg")
            img = expand2square(img, (255, 255, 255, 0))
            img.convert('RGB').save("rgb_pils_1.jpg")
            if img is not None:
                sparse_pils.append(img)
    img = Image.open(preview_path)
    img = remove(img, session=session)
    img = change_rgba_bg(img, "white")
    img = expand2square(img, (255, 255, 255, 0))
    sparse_pils.append(img)

    # if preview_img.size[0] <= 512:
    #     preview_img = run_sr_fast([preview_img])[0] #TODO
    # if sparse_pils[0].size[0] <= 512:
    #     sparse_pils = run_sr_fast(sparse_pils) #TODO
    
    
    rgb_pils, front_pil = run_mvprediction(preview_img, remove_bg=input_processing, seed=int(seed)) # 6s
    
    sparse_cameras = get_sparse_cameras_list_orth(
                                preview_path,
                                sparse_path,
                                raw_size = sparse_pils[0].size[0],
                                input_size = rgb_pils[0].size[0],
                                device="cuda")
    
    # sparse_cameras = get_sparse_cameras_list(
    #                             preview_path,
    #                             sparse_path,
    #                             '../data/snow_boy/',
    #                             raw_size = sparse_pils[0].size[0],
    #                             input_size = rgb_pils[0].size[0],
    #                             device="cuda")
    sparse_pils = [img.resize((512,512)) for img in sparse_pils]
    
    # sparse_pils = run_sr_fast(sparse_pils)
    sparse_pils = [change_rgba_bg(img, "white") for img in sparse_pils]
    # rgb_pils[0].convert('RGB').save("rgb_pils_0.jpg")
    sparse_pils[0].convert('RGB').save("sparse_pils_0.jpg")
    sparse_pils[1].convert('RGB').save("sparse_pils_1.jpg")
    new_meshes = geo_reconstruct(rgb_pils, None, sparse_pils, sparse_cameras, front_pil, do_refine=do_refine, predict_normal=True, expansion_weight=expansion_weight, init_type=init_type)
    vertices = new_meshes.verts_packed()
    vertices = vertices / 2 * 1.35
    vertices[..., [0, 2]] = - vertices[..., [0, 2]]
    new_meshes = Meshes(verts=[vertices], faces=new_meshes.faces_list(), textures=new_meshes.textures)
    
    ret_mesh, video = save_glb_and_video("/tmp/gradio/generated", new_meshes, with_timestamp=True, dist=3.5, fov_in_degrees=2 / 1.35, cam_type="ortho", export_video=render_video)
    return ret_mesh, video

#######################################
def create_ui(concurrency_id="wkl"):
    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(type='pil', image_mode='RGBA', label='Frontview')
            
            example_folder = os.path.join(os.path.dirname(__file__), "./examples")
            example_fns = sorted([os.path.join(example_folder, example) for example in os.listdir(example_folder)])
            gr.Examples(
                examples=example_fns,
                inputs=[input_image],
                cache_examples=False,
                label='Examples (click one of the images below to start)',
                examples_per_page=12
            )
            

        with gr.Column(scale=3):
            # export mesh display
            output_mesh = gr.Model3D(value=None, label="Mesh Model", show_label=True, height=320)
            output_video = gr.Video(label="Preview", show_label=True, show_share_button=True, height=320, visible=False)
            
            input_processing = gr.Checkbox(
                value=True,
                label='Remove Background',
                visible=True,
            )
            do_refine = gr.Checkbox(value=True, label="Refine Multiview Details", visible=False)
            expansion_weight = gr.Slider(minimum=-1., maximum=1.0, value=0.1, step=0.1, label="Expansion Weight", visible=False)
            init_type = gr.Dropdown(choices=["std", "thin"], label="Mesh Initialization", value="std", visible=False)
            setable_seed = gr.Slider(-1, 1000000000, -1, step=1, visible=True, label="Seed")
            render_video = gr.Checkbox(value=False, visible=False, label="generate video")
            fullrunv2_btn = gr.Button('Generate 3D', interactive=True)
            
    fullrunv2_btn.click(
        fn = generate3dv2,
        inputs=[input_image, input_processing, setable_seed, render_video, do_refine, expansion_weight, init_type],
        outputs=[output_mesh, output_video],
        concurrency_id=concurrency_id,
        api_name="generate3dv2",
    ).success(clean_up, api_name=False)
    return input_image
