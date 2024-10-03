import argparse
import glob
import json
import os
import pickle
import shutil

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from diffusers import AutoencoderKL
from face_alignment import NetworkSize
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from tqdm import tqdm

try:
    from utils.face_parsing import FaceParsing
except ModuleNotFoundError:
    from musetalk.utils.face_parsing import FaceParsing


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    """
    Extract frames from a video file and save them as images.

    :param vid_path: Path to the video file.
    :param save_path: Directory where extracted images will be saved.
    :param ext: Image file extension.
    :param cut_frame: Maximum number of frames to extract.
    """
    cap = cv2.VideoCapture(vid_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(frame_count, cut_frame)
    for count in tqdm(range(total_frames), desc="Extracting frames"):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(save_path, f"{count:08d}{ext}"), frame)
        else:
            break
    cap.release()


def read_imgs(img_list):
    """
    Read a list of image file paths into memory.

    :param img_list: List of image file paths.
    :return: List of image arrays.
    """
    print('Reading images...')
    frames = [cv2.imread(img_path) for img_path in tqdm(img_list, desc="Loading images")]
    return frames


def get_landmark_and_bbox(img_list, upperbondrange=0):
    """
    Get landmarks and bounding boxes for a list of images.

    :param img_list: List of image file paths.
    :param upperbondrange: Shift value for the upper boundary of the face bounding box.
    :return: Tuple of coordinates list and frames list.
    """
    frames = read_imgs(img_list)
    coords_list = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    print(f'Getting key landmarks and face bounding boxes with bbox_shift: {upperbondrange}')
    for frame in tqdm(frames, desc="Processing frames"):
        result = inference_topdown(model, frame)
        result = merge_data_samples(result)
        keypoints = result.pred_instances.keypoints
        if keypoints.shape[0] == 0:
            coords_list.append(coord_placeholder)
            continue
        face_landmarks = keypoints[0][23:91].astype(np.int32)

        # Get bounding boxes by face detection
        bbox = fa.get_detections_for_batch(np.expand_dims(frame, axis=0))

        if bbox[0] is None:
            coords_list.append(coord_placeholder)
            continue

        half_face_coord = face_landmarks[29].copy()
        if upperbondrange != 0:
            half_face_coord[1] += upperbondrange

        half_face_dist = np.max(face_landmarks[:, 1]) - half_face_coord[1]
        upper_bond = half_face_coord[1] - half_face_dist

        f_landmark = (
            np.min(face_landmarks[:, 0]),
            int(upper_bond),
            np.max(face_landmarks[:, 0]),
            np.max(face_landmarks[:, 1])
        )
        x1, y1, x2, y2 = f_landmark

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            coords_list.append(bbox[0])
            print("Error in bounding box:", bbox[0])
        else:
            coords_list.append(f_landmark)
    return coords_list, frames


class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)
        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True
            print('CUDA acceleration is enabled.')

        # Get the face detector
        face_detector_module = __import__('face_detection.detection.' + face_detector,
                                          fromlist=[face_detector])

        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

    def get_detections_for_batch(self, images):
        images_rgb = images[..., ::-1]  # Convert BGR to RGB
        detected_faces = self.face_detector.detect_from_batch(images_rgb.copy())
        results = [
            (int(d[0][0]), int(d[0][1]), int(d[0][2]), int(d[0][3])) if len(d) > 0 else None
            for d in detected_faces
        ]
        return results


def get_mask_tensor():
    """
    Creates a mask tensor for image processing.
    :return: A mask tensor.
    """
    mask_tensor = torch.zeros((256, 256))
    mask_tensor[:128, :] = 1
    return mask_tensor


def preprocess_img(img_input, half_mask=False):
    """
    Preprocess an image for model input.

    :param img_input: Image file path or image array.
    :param half_mask: Whether to apply a half mask to the image.
    :return: Preprocessed image tensor.
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    img = img / 255.0
    img_tensor = torch.FloatTensor(img).permute(2, 0, 1)
    if half_mask:
        mask = get_mask_tensor()
        img_tensor *= mask
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    img_tensor = normalize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor


def encode_latents(image):
    """
    Encode an image tensor into latent space.

    :param image: Image tensor.
    :return: Latent representation.
    """
    with torch.no_grad():
        init_latent_dist = vae.encode(image.to(vae.dtype)).latent_dist
    init_latents = vae.config.scaling_factor * init_latent_dist.sample()
    return init_latents


def get_latents_for_unet(img):
    """
    Get the latent inputs for the U-Net model.

    :param img: Image array.
    :return: Concatenated latent inputs.
    """
    ref_image_masked = preprocess_img(img, half_mask=True)
    masked_latents = encode_latents(ref_image_masked)
    ref_image_full = preprocess_img(img, half_mask=False)
    ref_latents = encode_latents(ref_image_full)
    latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
    return latent_model_input


def get_crop_box(box, expand):
    """
    Calculate a crop box expanded around a given bounding box.

    :param box: Original bounding box (x, y, x1, y1).
    :param expand: Expansion factor.
    :return: Expanded crop box and size.
    """
    x, y, x1, y1 = box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    s = int(max(x1 - x, y1 - y) * expand / 2)
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    return crop_box, s


def face_seg(image):
    """
    Perform face segmentation on an image.

    :param image: PIL Image.
    :return: Segmented image.
    """
    seg_image = fp(image)
    if seg_image is None:
        print("Error: No person segment found.")
        return None
    return seg_image.resize(image.size)


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.2):
    """
    Prepare materials for image processing.

    :param image: Image array.
    :param face_box: Face bounding box.
    :param upper_boundary_ratio: Ratio to keep upper boundary of the talking area.
    :param expand: Expansion factor for the crop box.
    :return: Mask array and crop box.
    """
    body = Image.fromarray(image[:, :, ::-1])

    crop_box, _ = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)

    mask_image = face_seg(face_large)
    if mask_image is None:
        mask_array = np.zeros((face_large.size[1], face_large.size[0]), dtype=np.uint8)
        return mask_array, crop_box

    x1, y1, x2, y2 = face_box
    mask_small = mask_image.crop((x1 - x_s, y1 - y_s, x2 - x_s, y2 - y_s))
    mask_image_full = Image.new('L', face_large.size, 0)
    mask_image_full.paste(mask_small, (x1 - x_s, y1 - y_s, x2 - x_s, y2 - y_s))

    width, height = mask_image_full.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', face_large.size, 0)
    modified_mask_image.paste(mask_image_full.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * face_large.size[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box


def is_video_file(file_path):
    """
    Check if a file is a video file based on its extension.

    :param file_path: Path to the file.
    :return: Boolean indicating if the file is a video.
    """
    video_exts = ['.mp4', '.mkv', '.flv', '.avi', '.mov']
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in video_exts


def create_dir(dir_path):
    """
    Create a directory if it doesn't exist.

    :param dir_path: Path to the directory.
    """
    os.makedirs(dir_path, exist_ok=True)


current_dir = os.path.dirname(os.path.abspath(__file__))


def create_musetalk_human(file, avatar_id):
    """
    Main function to create MuseTalk avatar materials.

    :param file: Path to the input video or image(s).
    :param avatar_id: Identifier for the avatar.
    """
    # Set up paths
    save_path = os.path.join(current_dir, f'../data/avatars/avator_{avatar_id}')
    save_full_path = os.path.join(save_path, 'full_imgs')
    create_dir(save_full_path)
    mask_out_path = os.path.join(save_path, 'mask')
    create_dir(mask_out_path)
    mask_coords_path = os.path.join(save_path, 'mask_coords.pkl')
    coords_path = os.path.join(save_path, 'coords.pkl')
    latents_out_path = os.path.join(save_path, 'latents.pt')

    # Save avatar info
    with open(os.path.join(save_path, 'avator_info.json'), "w") as f:
        json.dump({
            "avatar_id": avatar_id,
            "video_path": file,
            "bbox_shift": 5
        }, f)

    # Process input file
    if os.path.isfile(file):
        if is_video_file(file):
            video2imgs(file, save_full_path, ext='.png')
        else:
            shutil.copy(file, os.path.join(save_full_path, os.path.basename(file)))
    else:
        png_files = sorted(glob.glob(os.path.join(file, '*.png')))
        for src in png_files:
            dst = os.path.join(save_full_path, os.path.basename(src))
            shutil.copy(src, dst)

    input_img_list = sorted(glob.glob(os.path.join(save_full_path, '*.[jpJP][pnPN]*[gG]')))
    print("Extracting landmarks...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, 5)
    input_latent_list = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)

    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = get_latents_for_unet(resized_crop_frame)
        input_latent_list.append(latents)

    # Cycle frames for continuity
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    mask_coords_list_cycle = []

    for i, (frame, face_box) in enumerate(tqdm(zip(frame_list_cycle, coord_list_cycle), total=len(frame_list_cycle), desc="Processing masks")):
        cv2.imwrite(os.path.join(save_full_path, f"{str(i).zfill(8)}.png"), frame)
        mask, crop_box = get_image_prepare_material(frame, face_box)
        cv2.imwrite(os.path.join(mask_out_path, f"{str(i).zfill(8)}.png"), mask)
        mask_coords_list_cycle.append(crop_box)

    # Save processing results
    with open(mask_coords_path, 'wb') as f:
        pickle.dump(mask_coords_list_cycle, f)

    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list_cycle, f)

    torch.save(input_latent_list, latents_out_path)


# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(1, flip_input=False, device=device)
config_file = os.path.join(current_dir, 'utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py')
checkpoint_file = os.path.abspath(os.path.join(current_dir, '../models/dwpose/dw-ll_ucoco_384.pth'))
model = init_model(config_file, checkpoint_file, device=device)
vae = AutoencoderKL.from_pretrained(os.path.abspath(os.path.join(current_dir, '../models/sd-vae-ft-mse')))
vae.to(device)
fp = FaceParsing(os.path.abspath(os.path.join(current_dir, '../models/face-parse-bisent/resnet18-5c106cde.pth')),
                 os.path.abspath(os.path.join(current_dir, '../models/face-parse-bisent/79999_iter.pth')))

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",
                        type=str,
                        default=r'',
                        )
    parser.add_argument("--avatar_id",
                        type=str,
                        default='3',
                        )
    args = parser.parse_args()
    create_musetalk_human(args.file, args.avatar_id)
