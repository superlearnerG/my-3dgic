import re
import os
import sys
import glob
import json
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, read_next_bytes
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm
import torch

try:
    import pyexr
except Exception as e:
    print(e)
    # raise e
    pyexr = None


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    FovY: np.array = None
    FovX: np.array = None
    fx: np.array = None
    fy: np.array = None
    cx: np.array = None
    cy: np.array = None
    normal: np.array = None
    hdr: bool = False
    depth: np.array = None
    image_mask: np.array = None
    objects: np.array = None
    depths: np.array = None
    depth_loss_path: str = ""
    depth_loss_scale: float = 1.0


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_img(path):
    if not "." in os.path.basename(path):
        files = glob.glob(path + '.*')
        assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
        path = files[0]
    if path.endswith(".exr"):
        if pyexr is not None:
            exr_file = pyexr.open(path)
            # print(exr_file.channels)
            all_data = exr_file.get()
            img = all_data[..., 0:3]
            if "A" in exr_file.channels:
                mask = np.clip(all_data[..., 3:4], 0, 1)
                img = img * mask
        else:
            img = imageio.imread(path)
            import pdb;
            pdb.set_trace()
        img = np.nan_to_num(img)
        hdr = True
    else:  # LDR image
        img = imageio.imread(path)
        img = img / 255
        # img[..., 0:3] = srgb_to_rgb_np(img[..., 0:3])
        hdr = False
    return img, hdr


def load_pfm(file: str):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(br'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = data[::-1, ...]  # cv2.flip(data, 0)

    return np.ascontiguousarray(data)


def load_depth(tiff_path):
    return imageio.imread(tiff_path)


def load_mask(mask_file):
    mask = imageio.imread(mask_file, mode='L')
    mask = mask.astype(np.float32)
    mask[mask > 0.5] = 1.0

    return mask


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".TIF", ".TIFF")


def _split_name_keys(name):
    stripped = str(name).strip()
    if not stripped:
        return set()

    basename = os.path.basename(stripped)
    stem = Path(basename).stem
    return {stripped, basename, stem}


def _read_split_list(list_path):
    split_names = set()
    with open(list_path, "r") as file:
        for line in file:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            split_names.update(_split_name_keys(item))
    return split_names


def _name_in_split(image_name, split_names):
    return bool(_split_name_keys(image_name) & split_names)


def _image_index_from_name(image_name):
    stem = Path(os.path.basename(str(image_name))).stem
    try:
        return int(stem)
    except ValueError:
        match = re.search(r"(\d+)$", stem)
        if match:
            return int(match.group(1))
    return None


def _is_mod8_test_image(image_name):
    image_index = _image_index_from_name(image_name)
    return image_index is not None and image_index % 8 == 0


def _find_by_stem(folder, stem, extensions=IMAGE_EXTENSIONS):
    if not folder or not stem:
        return None

    folder = Path(folder)
    for ext in extensions:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)

    if folder.exists():
        matches = sorted(p for p in folder.iterdir() if p.is_file() and p.stem == stem)
        if matches:
            return str(matches[0])
    return None


def _first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _legacy_stem_candidates(image_stem, idx):
    stems = [image_stem]
    if "_" in image_stem:
        try:
            stems.append("{0:05d}".format(int(image_stem.split("_")[1]) - 1))
        except (ValueError, IndexError):
            pass
    stems.extend(["{0:05d}".format(idx), "frame_{0:05d}".format(idx + 1)])
    unique_stems = []
    for stem in stems:
        if stem not in unique_stems:
            unique_stems.append(stem)
    return unique_stems


def _find_sidecar(folder, stems, extensions=IMAGE_EXTENSIONS):
    for stem in stems:
        path = _find_by_stem(folder, stem, extensions)
        if path:
            return path
    return None


def _find_named_sidecar(folder, stems, suffix, extensions=("",)):
    for stem in stems:
        for ext in extensions:
            path = os.path.join(folder, f"{stem}{suffix}{ext}")
            if os.path.exists(path):
                return path
    return None


def _paired_suffix_path(path, old_suffix, new_suffix):
    path = Path(path)
    stem = path.stem
    if stem.endswith(old_suffix):
        return str(path.with_name(stem[:-len(old_suffix)] + new_suffix + path.suffix))
    return None


def _depth_array_2d(depth, depth_path):
    depth = np.asarray(depth)
    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
        else:
            raise ValueError(f"Expected a single-channel raw depth map at '{depth_path}', got shape {depth.shape}.")
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D raw depth map at '{depth_path}', got shape {depth.shape}.")
    return depth


def _resolve_depth_folder(path, depths, use_depth_loss):
    if not use_depth_loss:
        return ""
    depth_dir = depths if depths else "depth"
    depth_folder = depth_dir if os.path.isabs(depth_dir) else os.path.join(path, depth_dir)
    if not os.path.isdir(depth_folder):
        raise FileNotFoundError(f"--use_depth_loss expects raw .npy depth maps under '{depth_folder}'.")
    print(f"[Depth Loss] Loading raw depth maps from {depth_folder}")
    return depth_folder


def _raw_depth_path(depths_folder, image_name):
    if depths_folder == "":
        return ""
    return os.path.join(depths_folder, f"{Path(image_name).stem}.npy")


def _read_points3d_xyz_by_id_binary(path):
    points = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point_props = read_next_bytes(fid, 43, "QdddBBBd")
            point_id = int(point_props[0])
            points[point_id] = np.array(point_props[1:4], dtype=np.float64)
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(8 * track_length, 1)
    return points


def _read_points3d_xyz_by_id_text(path):
    points = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            points[int(elems[0])] = np.array(tuple(map(float, elems[1:4])), dtype=np.float64)
    return points


def _read_points3d_xyz_by_id(path):
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if os.path.exists(bin_path):
        try:
            return _read_points3d_xyz_by_id_binary(bin_path)
        except Exception:
            if not os.path.exists(txt_path):
                raise
    if os.path.exists(txt_path):
        return _read_points3d_xyz_by_id_text(txt_path)
    raise FileNotFoundError(f"COLMAP points3D file not found under {os.path.join(path, 'sparse/0')}")


def _select_evenly_spaced(items, max_count):
    if len(items) <= max_count:
        return items
    indices = np.linspace(0, len(items) - 1, max_count, dtype=int)
    return [items[int(idx)] for idx in indices]


def _estimate_depth_scale_from_colmap(path, cam_extrinsics, depths_folder, max_views=32, max_points_per_view=12000):
    xyz_by_id = _read_points3d_xyz_by_id(path)
    ratios = []
    used_views = 0

    extrinsics = sorted(cam_extrinsics.values(), key=lambda extr: extr.name)
    for extr in _select_evenly_spaced(extrinsics, max_views):
        depth_path = _raw_depth_path(depths_folder, extr.name)
        if not os.path.exists(depth_path):
            continue

        point_ids = np.asarray(extr.point3D_ids)
        xys = np.asarray(extr.xys)
        valid_indices = np.flatnonzero(point_ids != -1)
        if valid_indices.size == 0:
            continue
        if valid_indices.size > max_points_per_view:
            valid_indices = np.asarray(_select_evenly_spaced(valid_indices.tolist(), max_points_per_view))

        matched_xys = []
        matched_xyz = []
        for idx in valid_indices:
            xyz = xyz_by_id.get(int(point_ids[idx]))
            if xyz is None:
                continue
            matched_xys.append(xys[idx])
            matched_xyz.append(xyz)
        if not matched_xyz:
            continue

        raw_depth = _depth_array_2d(np.load(depth_path, mmap_mode="r"), depth_path)
        matched_xys = np.asarray(matched_xys, dtype=np.float64)
        matched_xyz = np.asarray(matched_xyz, dtype=np.float64)
        u = np.rint(matched_xys[:, 0]).astype(np.int64)
        v = np.rint(matched_xys[:, 1]).astype(np.int64)
        in_image = (u >= 0) & (v >= 0) & (u < raw_depth.shape[1]) & (v < raw_depth.shape[0])
        if not np.any(in_image):
            continue

        R = qvec2rotmat(extr.qvec)
        t = np.asarray(extr.tvec, dtype=np.float64)
        z_colmap = (R @ matched_xyz[in_image].T).T[:, 2] + t[2]
        raw_z = np.asarray(raw_depth[v[in_image], u[in_image]], dtype=np.float64)
        valid = np.isfinite(raw_z) & (raw_z > 0.0) & np.isfinite(z_colmap) & (z_colmap > 0.0)
        view_ratios = z_colmap[valid] / raw_z[valid]
        view_ratios = view_ratios[np.isfinite(view_ratios) & (view_ratios > 0.0) & (view_ratios < 100.0)]
        if view_ratios.size == 0:
            continue
        ratios.append(view_ratios)
        used_views += 1

    if not ratios:
        raise RuntimeError(
            "Unable to estimate --depth_scale from COLMAP tracks and raw depth maps. "
            "Pass a positive --depth_scale manually."
        )

    ratios = np.concatenate(ratios)
    if ratios.size < 100:
        raise RuntimeError(
            f"Only {ratios.size} valid COLMAP/raw-depth correspondences were found; "
            "pass a positive --depth_scale manually."
        )

    scale = float(np.median(ratios))
    print(
        "[Depth Loss] Estimated raw-depth scale from COLMAP tracks: "
        f"{scale:.6f} ({ratios.size} samples from {used_views} views; "
        f"p05={np.percentile(ratios, 5):.6f}, p95={np.percentile(ratios, 95):.6f})"
    )
    return scale


def _resolve_depth_scale(path, cam_extrinsics, depths_folder, requested_depth_scale, use_depth_loss):
    if not use_depth_loss:
        return 1.0
    requested_depth_scale = float(requested_depth_scale)
    if requested_depth_scale > 0.0:
        print(f"[Depth Loss] Using manual raw-depth scale: {requested_depth_scale:.6f}")
        return requested_depth_scale
    return _estimate_depth_scale_from_colmap(path, cam_extrinsics, depths_folder)


def _validate_train_depth_paths(train_cam_infos):
    missing = [cam.image_name for cam in train_cam_infos if not getattr(cam, "depth_loss_path", "")]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing)"
        raise FileNotFoundError(f"Missing raw .npy depth maps for training views: {preview}{suffix}")


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, objects_folder, debug=True,
                      depths_folder="", depth_scale=1.0):
    debug=True
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            ppx = intr.params[1]
            ppy = intr.params[2]

            Fovx = focal2fov(focal_length_x, width)
            FovY = focal2fov(focal_length_y, height)

        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            ppx = intr.params[2]
            ppy = intr.params[3]

            Fovx = focal2fov(focal_length_x, width)
            FovY = focal2fov(focal_length_y, height)

        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        # print(extr.name)

        image_basename = os.path.basename(extr.name)
        image_stem = Path(image_basename).stem
        depth_loss_path = _raw_depth_path(depths_folder, image_basename)
        if depth_loss_path and not os.path.isfile(depth_loss_path):
            depth_loss_path = ""
        image_path = os.path.join(images_folder, image_basename)
        image_name = image_stem
        # image, is_hdr = load_img(os.path.join(images_folder, image_name))
        # print(image)
        # input()
        # object_path = os.path.join(objects_folder, image_name + '.png')
        
        # try:
        #     objects = Image.open(object_path)
        # except:
        #     import ipdb; ipdb.set_trace()
        # try:
        #     mask_path = os.path.join(os.path.dirname(images_folder), "object_mask", os.path.basename(extr.name))
        #     print(mask_path)
        #     input()
        #     mask = np.array(Image.open(mask_path), dtype=np.float32) / 255
        # except:
        #     mask_path = os.path.join(os.path.dirname(images_folder), "inpaint_object_mask_255", image_name + '.png')
        #     mask = np.array(Image.open(mask_path), dtype=np.float32) / 255
        
        
        # add depth for COLMAP dataset
        gt_depth = None
        gt_depth_path = _first_existing([
            os.path.join(os.path.dirname(images_folder), "filtered/depths", image_stem + ".tiff"),
            os.path.join(os.path.dirname(images_folder), "filtered/depths", image_basename.replace(".png", ".tiff")),
        ])
        if gt_depth_path and os.path.exists(gt_depth_path):
            gt_depth = load_depth(gt_depth_path)

        image_path = _first_existing([os.path.join(images_folder, image_basename)])
        if image_path is None:
            image_path = _find_by_stem(images_folder, image_stem)
        if image_path is None and Path(images_folder).name != "substitude":
            image_path = _find_sidecar(images_folder, ["frame_{0:05d}".format(idx + 1), "{0:05d}".format(idx)])
        # print(image_path)
        # print(images_folder)
        depth_folder = (os.path.join(objects_folder,"depth_removal"))
        original_obj_folder = (os.path.join(objects_folder,"obj_original"))
        gt_folder = (os.path.join(objects_folder,"images"))
        
        image_name = Path(image_path).stem if image_path is not None else image_stem
        sidecar_stems = _legacy_stem_candidates(image_name, idx)
        if image_stem not in sidecar_stems:
            sidecar_stems.insert(1, image_stem)
        # print(image_name)
        # input()
        # input()
        # print(image_path)
        image = Image.open(image_path) if image_path is not None and os.path.exists(image_path) else None

        if image is not None:
            image = np.asarray(image).astype(np.float32) / 255.0
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            mask = np.ones_like(image[..., 0])
            if image.shape[-1] == 4:
                alpha = image[..., 3]
                bg = np.zeros(3, dtype=image.dtype)
                image = image[..., :3] * alpha[..., None] + bg * (1.0 - alpha[..., None])
                mask = alpha
            elif image.shape[-1] > 3:
                image = image[..., :3]
        else:
            mask = None

        # print(image_path)
        # print(image)

        # print(image)
        # input()
        id_name = sidecar_stems[0]

        # print(id_name)
        
        object_path = _find_sidecar(objects_folder, sidecar_stems)
        # print(object_path)
        # input()
        objects = Image.open(object_path).convert('L') if object_path and os.path.exists(object_path) else None
        # print(object_path)
        # print(objects)
        # depth_path = os.path.join(depth_folder, id_name + '.pt')
        # print(os.path.basename(objects_folder))
        if os.path.basename(objects_folder) == 'object_mask':
            # print("suceed")
            depth_path = _find_named_sidecar(depth_folder, sidecar_stems, "_original", extensions=(".pt",))
            depths = torch.load(depth_path).cpu() if depth_path and os.path.exists(depth_path) else None
            # print(depth_path)
            # print(depth_path)

            # print(depths)
            #     exit()
            # input()
            if depths is not None:
                depths = depths.repeat(3,1,1).permute(1,2,0).numpy()
                depth_remove_path = _paired_suffix_path(depth_path, "_original", "_remove")
                if depth_remove_path is None or not os.path.exists(depth_remove_path):
                    depth_remove_path = _find_named_sidecar(depth_folder, sidecar_stems, "_remove", extensions=(".pt",))
                if depth_remove_path and os.path.exists(depth_remove_path):
                    depth_remove = torch.load(depth_remove_path).cpu().squeeze()
                    depths[:,:,1] = depth_remove

                # depths = depths.numpy()
                # input()


            depth_range = None
        else:
            depth_path = _find_sidecar(depth_folder, sidecar_stems)
            # print(depth_path)
            depths = Image.open(depth_path) if depth_path and os.path.exists(depth_path) else None
            depth_range_path = None
            if depth_path:
                depth_range_path = str(Path(depth_path).with_name(Path(depth_path).stem + "_range.pt"))
            if not depth_range_path or not os.path.exists(depth_range_path):
                depth_range_path = _find_named_sidecar(depth_folder, sidecar_stems, "_range", extensions=(".pt",))
            # print(depth_range_path)
            depth_range = torch.load(depth_range_path) if depth_range_path and os.path.exists(depth_range_path) else None
        # print(depth_path)
        # print(id_name, image_name)
        gt_image_path = _find_sidecar(gt_folder, sidecar_stems)
        gt_image = Image.open(gt_image_path) if gt_image_path and os.path.exists(gt_image_path) else None
        original_obj_path = _find_sidecar(original_obj_folder, sidecar_stems)
        original_obj = Image.open(original_obj_path) if original_obj_path and os.path.exists(original_obj_path) else None

        


        
            

        # print(original_obj_path, gt_image_path)
        # print(object_path)
        # print(depth_path)
        # print(depth_range_path)
        # print(image_path)
        # input()
        # print(image_path)
        # print(object_path)
        # print(depth_range_path)
        # print(depth_path)
        # input()
        # depths = depths/255.0
        # print(depths, depth_range)
        if (depths is not None) and (depth_range != None):

            # print(depth_range)
            # print(depths)
            # input()
            depths = (np.asarray(depths)-4)/255.0
            (near, far) = depth_range
            near, far = near.cpu().numpy(), far.cpu().numpy()
            depths = (depths)*(far-near)+near
            # print(depths)
        if (gt_image != None) and (original_obj != None):
            # image = [image, gt_image]
            # objects = [objects, original_obj]
            gt_image  = np.asarray(gt_image)/255.0
            # print(np.asarray(original_obj).shape)
            original_obj = np.asarray(original_obj)
            # print(depths.shape, gt_image.shape, original_obj.shape)
            # print(depths.shape, gt_image.shape)
            # print(original_obj.shape)
            try:
                depths = np.concatenate((np.concatenate((depths, gt_image), axis = 2), np.expand_dims(original_obj,2)), axis = 2).astype(np.float16)
            except:
                depths = depths
            # depths = np.array([depths, gt_image, original_obj])
            # print(depths.shape)
            # input()
            # depths = [depths, gt_image, original_obj]
        # depths = torch.load(depth_path) if os.path.exists(depth_path) else None
        # print(depth_path)
        # print(object_path)
        # print(image_path)
        # print(depths)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovX=Fovx, FovY=FovY, fx=focal_length_x, fy=focal_length_y, cx=ppx,
                              cy=ppy, image=(image.astype(np.float16) if image is not None else None), depth=gt_depth, image_mask=mask, objects=objects,
                              image_path=image_path, image_name=image_name, width=width, height=height, hdr=False, depths=depths,
                              depth_loss_path=depth_loss_path, depth_loss_scale=depth_scale)
        # if "test_" not in image_name:
        #     cam_infos.append(cam_info)
        # else:
        #     print("excluding image {}".format(image_name))
        
        if image is not None:
            cam_infos.append(cam_info)
        else:
            print("excluding image {}".format(extr.name))


        # if debug and idx >= 5:
        #     break
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T

    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32)
        colors /= 255.0

    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    if np.all(normals == 0):
        print("random init normal")
        normals = np.random.random(normals.shape)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if normals is None:
        normals = np.random.randn(*xyz.shape)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, object_path, llffhold=8, debug=False,
                        use_depth_loss=False, depths="", depth_scale=0.0):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images
    object_dir = 'object_mask' if object_path == None else object_path
    depths_folder = _resolve_depth_folder(path, depths, use_depth_loss)
    resolved_depth_scale = _resolve_depth_scale(path, cam_extrinsics, depths_folder, depth_scale, use_depth_loss)
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir),
                                           objects_folder=os.path.join(path, object_dir),
                                           debug=debug,
                                           depths_folder=depths_folder,
                                           depth_scale=resolved_depth_scale)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_list_path = os.path.join(path, "train_list.txt")
        test_list_path = os.path.join(path, "test_list.txt")
        if os.path.exists(train_list_path) and os.path.exists(test_list_path):
            print("COLMAP split: using train_list.txt and test_list.txt")
            train_split_names = _read_split_list(train_list_path)
            test_split_names = _read_split_list(test_list_path)
            train_cam_infos = [c for c in cam_infos if _name_in_split(c.image_name, train_split_names)]
            test_cam_infos = [c for c in cam_infos if _name_in_split(c.image_name, test_split_names)]
            if not train_cam_infos:
                raise ValueError(f"No COLMAP cameras matched {train_list_path}.")
            if not test_cam_infos:
                raise ValueError(f"No COLMAP cameras matched {test_list_path}.")
        else:
            print("COLMAP split: train_list.txt/test_list.txt not found; using basename index % 8 == 0 as test")
            test_cam_infos = [c for c in cam_infos if _is_mod8_test_image(c.image_name)]
            train_cam_infos = [c for c in cam_infos if not _is_mod8_test_image(c.image_name)]
            if not test_cam_infos:
                print("Warning: no numeric basename matched index % 8 == 0; test set is empty.")
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if not train_cam_infos:
        raise ValueError("No COLMAP training cameras after train/test split.")
    if use_depth_loss:
        _validate_train_depth_paths(train_cam_infos)
    if eval:
        print(f"COLMAP split: {len(train_cam_infos)} train cameras, {len(test_cam_infos)} test cameras")
    else:
        print(f"COLMAP split disabled: {len(train_cam_infos)} train cameras, {len(test_cam_infos)} test cameras")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", debug=False):
    cam_infos = []

    read_mvs = False
    mvs_dir = f"{path}/extra"
    if os.path.exists(mvs_dir) and "train" not in transformsfile:
        print("Loading mvs as geometry constraint.")
        read_mvs = True

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            image_path = os.path.join(path, frame["file_path"] + extension)
            image_name = Path(image_path).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image, is_hdr = load_img(image_path)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            image_mask = np.ones_like(image[..., 0])
            if image.shape[-1] == 4:
                image_mask = image[:, :, 3]
                image = image[:, :, :3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])

            # read depth and mask
            depth = None
            normal = None
            if read_mvs:
                depth_path = os.path.join(mvs_dir + "/depths/", os.path.basename(frame["file_path"]) + ".tiff")
                normal_path = os.path.join(mvs_dir + "/normals/", os.path.basename(frame["file_path"]) + ".pfm")

                depth = load_depth(depth_path)
                normal = load_pfm(normal_path)

                depth = depth * image_mask
                normal = normal * image_mask[..., np.newaxis]

            fovy = focal2fov(fov2focal(fovx, image.shape[0]), image.shape[1])
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image, image_mask=image_mask,
                                        image_path=image_path, depth=depth, normal=normal, image_name=image_name,
                                        width=image.shape[1], height=image.shape[0], hdr=is_hdr))

            if debug and idx >= 5:
                break

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png", debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, debug=debug)
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension,
                                                   debug=debug)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        normals = np.random.randn(*xyz.shape)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        storePly(ply_path, xyz, SH2RGB(shs) * 255, normals)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info


def loadCamsFromScene(path, valid_list, white_background, debug):
    with open(f'{path}/sfm_scene.json') as f:
        sfm_scene = json.load(f)

    # load bbox transform
    bbox_transform = np.array(sfm_scene['bbox']['transform']).reshape(4, 4)
    bbox_transform = bbox_transform.copy()
    bbox_transform[[0, 1, 2], [0, 1, 2]] = bbox_transform[[0, 1, 2], [0, 1, 2]].max() / 2
    bbox_inv = np.linalg.inv(bbox_transform)

    # meta info
    image_list = sfm_scene['image_path']['file_paths']

    # camera parameters
    train_cam_infos = []
    test_cam_infos = []
    camera_info_list = sfm_scene['camera_track_map']['images']
    for i, (index, camera_info) in enumerate(camera_info_list.items()):
        # flg == 2 stands for valid camera 
        if camera_info['flg'] == 2:
            intrinsic = np.zeros((4, 4))
            intrinsic[0, 0] = camera_info['camera']['intrinsic']['focal'][0]
            intrinsic[1, 1] = camera_info['camera']['intrinsic']['focal'][1]
            intrinsic[0, 2] = camera_info['camera']['intrinsic']['ppt'][0]
            intrinsic[1, 2] = camera_info['camera']['intrinsic']['ppt'][1]
            intrinsic[2, 2] = intrinsic[3, 3] = 1

            extrinsic = np.array(camera_info['camera']['extrinsic']).reshape(4, 4)
            c2w = np.linalg.inv(extrinsic)
            c2w[:3, 3] = (c2w[:4, 3] @ bbox_inv.T)[:3]
            extrinsic = np.linalg.inv(c2w)

            R = np.transpose(extrinsic[:3, :3])
            T = extrinsic[:3, 3]

            focal_length_x = camera_info['camera']['intrinsic']['focal'][0]
            focal_length_y = camera_info['camera']['intrinsic']['focal'][1]
            ppx = camera_info['camera']['intrinsic']['ppt'][0]
            ppy = camera_info['camera']['intrinsic']['ppt'][1]

            image_path = os.path.join(path, image_list[index])
            image_name = Path(image_path).stem

            image, is_hdr = load_img(image_path)

            depth_path = os.path.join(path + "/depths/", os.path.basename(
                image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".tiff"))

            if os.path.exists(depth_path):
                depth = load_depth(depth_path)
                depth *= bbox_inv[0, 0]
            else:
                print("No depth map for test view.")
                depth = None

            normal_path = os.path.join(path + "/normals/", os.path.basename(
                image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".pfm"))
            if os.path.exists(normal_path):
                normal = load_pfm(normal_path)
            else:
                print("No normal map for test view.")
                normal = None

            mask_path = os.path.join(path + "/pmasks/", os.path.basename(
                image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".png"))
            if os.path.exists(mask_path):
                img_mask = (imageio.imread(mask_path, pilmode='L') > 0.1).astype(np.float32)
                # if pmask is available, mask the image for PSNR
                image *= img_mask[..., np.newaxis]
            else:
                img_mask = np.ones_like(image[:, :, 0])

            fovx = focal2fov(focal_length_x, image.shape[1])
            fovy = focal2fov(focal_length_y, image.shape[0])
            if int(index) in valid_list:
                image *= img_mask[..., np.newaxis]
                test_cam_infos.append(CameraInfo(uid=index, R=R, T=T, FovY=fovy, FovX=fovx, fx=focal_length_x,
                                                 fy=focal_length_y, cx=ppx, cy=ppy, image=image,
                                                 image_path=image_path, image_name=image_name,
                                                 depth=depth, image_mask=img_mask, normal=normal,
                                                 width=image.shape[1], height=image.shape[0], hdr=is_hdr))
            else:
                image *= img_mask[..., np.newaxis]
                depth *= img_mask
                normal *= img_mask[..., np.newaxis]

                train_cam_infos.append(CameraInfo(uid=index, R=R, T=T, FovY=fovy, FovX=fovx, fx=focal_length_x,
                                                  fy=focal_length_y, cx=ppx, cy=ppy, image=image,
                                                  image_path=image_path, image_name=image_name,
                                                  depth=depth, image_mask=img_mask, normal=normal,
                                                  width=image.shape[1], height=image.shape[0], hdr=is_hdr))
        if debug and i >= 5:
            break

    return train_cam_infos, test_cam_infos, bbox_transform


def readNeILFInfo(path, white_background, eval, debug=False):
    validation_indexes = []
    if eval:
        if "data_dtu" in path:
            validation_indexes = [2, 12, 17, 30, 34]
        else:
            raise NotImplementedError

    print("Reading Training transforms")
    if eval:
        print("Reading Test transforms")

    train_cam_infos, test_cam_infos, bbx_trans = loadCamsFromScene(
        f'{path}/inputs', validation_indexes, white_background, debug)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = f'{path}/inputs/model/sparse_bbx_scale.ply'
    if not os.path.exists(ply_path):
        org_ply_path = f'{path}/inputs/model/sparse.ply'

        # scale sparse.ply
        pcd = fetchPly(org_ply_path)
        inv_scale_mat = np.linalg.inv(bbx_trans)  # [4, 4]
        points = pcd.points
        xyz = (np.concatenate([points, np.ones_like(points[:, :1])], axis=-1) @ inv_scale_mat.T)[:, :3]
        normals = pcd.normals
        colors = pcd.colors

        storePly(ply_path, xyz, colors * 255, normals)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "NeILF": readNeILFInfo,
}
