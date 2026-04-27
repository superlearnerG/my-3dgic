import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
WORKFLOW_DIRNAME = "iterative_3dgic"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".TIF", ".TIFF"}


def parse_args():
    parser = argparse.ArgumentParser(description="Round-based multi-object inpainting wrapper for 3DGIC.")
    parser.add_argument("command", choices=("init", "run-round", "run-all", "status"))
    parser.add_argument("-m", "--model_path", required=True, help="Base 3DGIC model path containing iterative_3dgic.")
    parser.add_argument("-s", "--source_path", help="Original 3DGIC dataset path. Required for init.")
    parser.add_argument("--target_ids", nargs="+", type=int, help="Object ids to process in order. Required for init.")
    parser.add_argument("--round_index", type=int, default=0, help="Round index for run-round.")
    parser.add_argument("--base_model_path", help="Optional trained 3DGIC model path. Defaults to --model_path.")
    parser.add_argument("--removal_config_template", default=str(PROJECT_ROOT / "configs/object_removal/bear.json"))
    parser.add_argument("--inpaint_config_template", default=str(PROJECT_ROOT / "configs/object_inpaint/bear_new.json"))
    parser.add_argument("--top_k_ref_views", type=int, default=3)
    parser.add_argument("--simple_lama_device", default="cuda")
    parser.add_argument("--mask_threshold", type=int, default=0)
    parser.add_argument("--mask_dilation", type=int, default=0)
    parser.add_argument("--background_id", type=int, default=0)
    parser.add_argument("--finetune_iteration", type=int, default=None)
    parser.add_argument("--type", choices=("render", "neilf"), default="render")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing round working directories.")
    return parser.parse_args()


def read_json(path):
    with open(path, "r") as file:
        return json.load(file)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def workflow_root(model_path):
    return Path(model_path).expanduser().resolve() / WORKFLOW_DIRNAME


def round_paths(root, round_index):
    round_dir = root / f"round_{round_index:03d}"
    return {
        "round_dir": round_dir,
        "scene_in": round_dir / "scene_in",
        "scene_out": round_dir / "scene_out",
        "data_work": round_dir / "data_work",
        "removal_config": round_dir / "object_removal.json",
        "inpaint_config": round_dir / "object_inpaint.json",
        "meta": round_dir / "round_meta.json",
    }


def load_workflow(model_path):
    path = workflow_root(model_path) / "workflow.json"
    if not path.is_file():
        raise FileNotFoundError(f"Workflow not initialized: {path}")
    return read_json(path)


def save_meta(paths, **updates):
    meta_path = paths["meta"]
    if meta_path.is_file():
        meta = read_json(meta_path)
    else:
        meta = {}
    meta.update(updates)
    meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(meta_path, meta)
    return meta


def reset_dir(path, overwrite):
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing path without --overwrite: {path}")
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_or_link(src, dst):
    src = Path(src).expanduser().resolve()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def copy_dir_contents(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"Required directory not found: {src}")
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def parse_iteration_dir(path):
    name = Path(path).name
    if not name.startswith("iteration_"):
        return None
    try:
        return int(name[len("iteration_"):])
    except ValueError:
        return None


def latest_iteration_dir(model_path):
    point_cloud = Path(model_path) / "point_cloud"
    candidates = []
    if point_cloud.is_dir():
        for item in point_cloud.iterdir():
            iteration = parse_iteration_dir(item)
            if iteration is not None and item.is_dir():
                candidates.append((iteration, item))
    if not candidates:
        raise FileNotFoundError(f"No point_cloud/iteration_* directory found in {model_path}")
    return sorted(candidates, key=lambda item: item[0])[-1]


def promote_model(src_model, dst_model, overwrite, ply_src=None, classifier_src=None):
    src_model = Path(src_model).expanduser().resolve()
    dst_model = Path(dst_model)
    reset_dir(dst_model, overwrite)

    if ply_src is None:
        _, src_iter_dir = latest_iteration_dir(src_model)
        ply_src = src_iter_dir / "point_cloud.ply"
        classifier_src = classifier_src or src_iter_dir / "classifier.pth"
    else:
        ply_src = Path(ply_src)
        classifier_src = Path(classifier_src) if classifier_src is not None else None

    if not Path(ply_src).is_file():
        raise FileNotFoundError(f"Point cloud not found: {ply_src}")
    if classifier_src is None or not Path(classifier_src).is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_src}")

    dst_iter = dst_model / "point_cloud" / "iteration_0"
    dst_iter.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply_src, dst_iter / "point_cloud.ply")
    shutil.copy2(classifier_src, dst_iter / "classifier.pth")

    cfg_src = src_model / "cfg_args"
    if cfg_src.is_file():
        shutil.copy2(cfg_src, dst_model / "cfg_args")
    else:
        (dst_model / "cfg_args").write_text("Namespace()\n")

    cameras_src = src_model / "cameras.json"
    if cameras_src.is_file():
        shutil.copy2(cameras_src, dst_model / "cameras.json")


def prepare_data_work(source_path, data_work, completed_ids, background_id, overwrite):
    source_path = Path(source_path).expanduser().resolve()
    data_work = Path(data_work)
    reset_dir(data_work, overwrite)

    skip_names = {"object_mask", "inpaint_2d_unseen_mask", "substitude", "intersect_mask"}
    for item in source_path.iterdir():
        if item.name in skip_names:
            continue
        copy_or_link(item, data_work / item.name)

    src_masks = source_path / "object_mask"
    dst_masks = data_work / "object_mask"
    if not src_masks.is_dir():
        raise FileNotFoundError(f"object_mask directory not found: {src_masks}")
    dst_masks.mkdir(parents=True, exist_ok=True)

    completed_ids = [int(item) for item in completed_ids]
    for src in src_masks.rglob("*"):
        rel = src.relative_to(src_masks)
        dst = dst_masks / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in IMAGE_EXTENSIONS:
            with Image.open(src) as image:
                mask = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            for object_id in completed_ids:
                mask[mask == object_id] = background_id
            Image.fromarray(mask, mode="L").save(dst)
        else:
            shutil.copy2(src, dst)


def run_command(cmd):
    print("[iterative_3dgic]", " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], cwd=PROJECT_ROOT, check=True)


def find_removal_output(scene_in):
    root = Path(scene_in) / "train" / "ours_object_removal"
    candidates = []
    if root.is_dir():
        for item in root.iterdir():
            iteration = parse_iteration_dir(item)
            if iteration is not None and item.is_dir():
                candidates.append((iteration, item))
    if not candidates:
        raise FileNotFoundError(f"No removal render output found in {root}")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def organize_inpaint_inputs(removal_output, data_work):
    removal_output = Path(removal_output)
    inpaint_root = Path(data_work) / "inpaint_2d_unseen_mask"
    if inpaint_root.exists():
        shutil.rmtree(inpaint_root)
    inpaint_root.mkdir(parents=True, exist_ok=True)

    copy_dir_contents(removal_output / "inpaint_mask_pred", inpaint_root)
    copy_dir_contents(removal_output / "renders", inpaint_root / "images")
    copy_dir_contents(removal_output / "depth_removal", inpaint_root / "depth_removal")
    copy_dir_contents(removal_output / "gt_objects", inpaint_root / "obj_original")
    return inpaint_root


def copy_removal_depth_to_object_mask(removal_output, data_work):
    src = Path(removal_output) / "depth"
    dst = Path(data_work) / "object_mask" / "depth_removal"
    copy_dir_contents(src, dst)


def load_view_order(removal_output):
    path = Path(removal_output) / "view_order.json"
    if not path.is_file():
        return {}
    entries = read_json(path)
    return {str(entry.get("stem")): entry for entry in entries}


def select_reference_views(inpaint_root, removal_output, top_k, mask_threshold):
    inpaint_root = Path(inpaint_root)
    areas = []
    for mask_path in sorted(inpaint_root.glob("*.png")):
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        area = int((mask > mask_threshold).sum())
        if area > 0:
            areas.append((area, mask_path.stem, str(mask_path)))

    if not areas:
        raise RuntimeError(f"No non-empty inpaint masks found in {inpaint_root}")

    areas.sort(key=lambda item: (-item[0], item[1]))
    selected = areas[:top_k]
    view_order = load_view_order(removal_output)
    refs = []
    for area, stem, mask_path in selected:
        entry = view_order.get(stem, {})
        refs.append({
            "stem": stem,
            "area": area,
            "mask_path": mask_path,
            "ref_id": entry.get("index"),
            "ref_name": entry.get("camera_image_name") or entry.get("image_name") or stem,
        })
    return refs


def load_template(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config template not found: {path}")
    return read_json(path)


def write_round_configs(workflow, paths, target_id, refs, round_index):
    removal_config = load_template(workflow["removal_config_template"])
    removal_config.update({
        "select_obj_id": [target_id],
        "images": "images",
        "object_path": "object_mask",
        "source_path": str(paths["data_work"]),
    })
    write_json(paths["removal_config"], removal_config)

    exp_setting = f"iter3dgic_round_{round_index:03d}_obj_{target_id}"
    inpaint_config = load_template(workflow["inpaint_config_template"])
    inpaint_config.update({
        "select_obj_id": [target_id],
        "images": "substitude",
        "object_path": "inpaint_2d_unseen_mask",
        "source_path": str(paths["data_work"]),
        "use_ref": 1,
        "ref_names": [item["ref_name"] for item in refs],
        "ref_id": [item["ref_id"] for item in refs if item.get("ref_id") is not None],
        "finetune_iteration": workflow["finetune_iteration"],
        "exp_setting": exp_setting,
    })
    write_json(paths["inpaint_config"], inpaint_config)
    return exp_setting


def copy_substitude_images(data_work):
    inpaint_images = Path(data_work) / "inpaint_2d_unseen_mask" / "images"
    substitude = Path(data_work) / "substitude"
    copy_dir_contents(inpaint_images, substitude)


def run_simple_lama(workflow, inpaint_root, refs):
    stems = [item["stem"] for item in refs]
    cmd = [
        sys.executable,
        "tools/simple_lama_3dgic.py",
        "--inpaint_dir",
        inpaint_root,
        "--stems",
        *stems,
        "--device",
        workflow["simple_lama_device"],
        "--mask_dilation",
        workflow["mask_dilation"],
        "--mask_threshold",
        workflow["mask_threshold"],
        "--mode",
        "both",
    ]
    run_command(cmd)


def promote_round_output(paths, exp_setting, finetune_iteration, overwrite):
    final_iteration = int(finetune_iteration) - 1
    ply_src = (
        paths["scene_in"]
        / "point_cloud_object_inpaint"
        / f"{exp_setting}_iteration_{final_iteration}"
        / "point_cloud.ply"
    )
    classifier_src = paths["scene_in"] / "point_cloud" / "iteration_0" / "classifier.pth"
    promote_model(paths["scene_in"], paths["scene_out"], overwrite, ply_src=ply_src, classifier_src=classifier_src)


def command_init(args):
    if args.type != "render":
        raise ValueError("iterative_3dgic v1 only supports --type render")
    if not args.source_path:
        raise ValueError("--source_path is required for init")
    if not args.target_ids:
        raise ValueError("--target_ids is required for init")

    root = workflow_root(args.model_path)
    if root.exists() and not args.overwrite:
        raise FileExistsError(f"Workflow already exists. Use --overwrite to replace: {root}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    inpaint_template = load_template(args.inpaint_config_template)
    finetune_iteration = args.finetune_iteration or int(inpaint_template.get("finetune_iteration", 10000))
    workflow = {
        "base_model_path": str(Path(args.base_model_path or args.model_path).expanduser().resolve()),
        "source_path": str(Path(args.source_path).expanduser().resolve()),
        "target_ids": [int(item) for item in args.target_ids],
        "top_k_ref_views": int(args.top_k_ref_views),
        "simple_lama_device": args.simple_lama_device,
        "mask_threshold": int(args.mask_threshold),
        "mask_dilation": int(args.mask_dilation),
        "background_id": int(args.background_id),
        "removal_config_template": str(Path(args.removal_config_template).expanduser().resolve()),
        "inpaint_config_template": str(Path(args.inpaint_config_template).expanduser().resolve()),
        "finetune_iteration": int(finetune_iteration),
        "type": args.type,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(root / "workflow.json", workflow)
    for round_index, target_id in enumerate(workflow["target_ids"]):
        paths = round_paths(root, round_index)
        paths["round_dir"].mkdir(parents=True, exist_ok=True)
        save_meta(
            paths,
            status="initialized",
            round_index=round_index,
            target_id=target_id,
            completed_before=workflow["target_ids"][:round_index],
        )
    print(f"Initialized workflow: {root / 'workflow.json'}")


def command_run_round(args, workflow=None, round_index=None):
    workflow = workflow or load_workflow(args.model_path)
    if workflow.get("type", "render") != "render":
        raise ValueError("iterative_3dgic v1 only supports render workflows")

    root = workflow_root(args.model_path)
    round_index = args.round_index if round_index is None else round_index
    target_ids = workflow["target_ids"]
    if round_index < 0 or round_index >= len(target_ids):
        raise IndexError(f"round_index={round_index} is outside target_ids range")

    target_id = int(target_ids[round_index])
    paths = round_paths(root, round_index)
    paths["round_dir"].mkdir(parents=True, exist_ok=True)
    completed_before = [int(item) for item in target_ids[:round_index]]
    save_meta(paths, status="running", round_index=round_index, target_id=target_id, completed_before=completed_before)

    src_model = Path(workflow["base_model_path"]) if round_index == 0 else round_paths(root, round_index - 1)["scene_out"]
    if round_index > 0 and not src_model.is_dir():
        raise FileNotFoundError(f"Previous round scene_out not found: {src_model}")

    promote_model(src_model, paths["scene_in"], args.overwrite)
    save_meta(paths, status="scene_in_ready", scene_in=str(paths["scene_in"]))

    prepare_data_work(
        workflow["source_path"],
        paths["data_work"],
        completed_before,
        workflow["background_id"],
        args.overwrite,
    )
    save_meta(paths, status="data_work_ready", data_work=str(paths["data_work"]))

    removal_config = load_template(workflow["removal_config_template"])
    removal_config.update({
        "select_obj_id": [target_id],
        "images": "images",
        "object_path": "object_mask",
        "source_path": str(paths["data_work"]),
    })
    write_json(paths["removal_config"], removal_config)

    run_command([sys.executable, "edit_object_removal.py", "-m", paths["scene_in"], "--config_file", paths["removal_config"], "--skip_test"])
    removal_output = find_removal_output(paths["scene_in"])
    copy_removal_depth_to_object_mask(removal_output, paths["data_work"])
    save_meta(paths, status="removal_depth_ready", removal_output=str(removal_output))

    run_command([
        sys.executable,
        "edit_object_removal.py",
        "-m",
        paths["scene_in"],
        "--config_file",
        paths["removal_config"],
        "--skip_test",
        "--render_intersect",
    ])
    removal_output = find_removal_output(paths["scene_in"])
    inpaint_root = organize_inpaint_inputs(removal_output, paths["data_work"])
    refs = select_reference_views(
        inpaint_root,
        removal_output,
        int(workflow["top_k_ref_views"]),
        int(workflow["mask_threshold"]),
    )
    exp_setting = write_round_configs(workflow, paths, target_id, refs, round_index)
    save_meta(paths, status="refs_selected", selected_refs=refs, exp_setting=exp_setting)

    try:
        run_simple_lama(workflow, inpaint_root, refs)
    except Exception:
        save_meta(paths, status="lama_failed")
        raise
    save_meta(paths, status="lama_outputs_ready")

    copy_substitude_images(paths["data_work"])
    run_command([sys.executable, "edit_object_inpaint_spin.py", "-m", paths["scene_in"], "--config_file", paths["inpaint_config"], "--skip_test"])
    save_meta(paths, status="inpaint_finished")

    promote_round_output(paths, exp_setting, workflow["finetune_iteration"], args.overwrite)
    save_meta(paths, status="completed", scene_out=str(paths["scene_out"]))


def command_run_all(args):
    workflow = load_workflow(args.model_path)
    for round_index in range(len(workflow["target_ids"])):
        command_run_round(args, workflow=workflow, round_index=round_index)


def command_status(args):
    workflow = load_workflow(args.model_path)
    root = workflow_root(args.model_path)
    print(f"workflow: {root / 'workflow.json'}")
    print(f"target_ids: {workflow['target_ids']}")
    for round_index, target_id in enumerate(workflow["target_ids"]):
        meta_path = round_paths(root, round_index)["meta"]
        if meta_path.is_file():
            meta = read_json(meta_path)
            status = meta.get("status", "unknown")
            selected = meta.get("selected_refs", [])
            refs = ",".join(str(item.get("stem")) for item in selected)
        else:
            status = "missing"
            refs = ""
        print(f"round_{round_index:03d} target={target_id} status={status} refs={refs}")


def main():
    args = parse_args()
    if args.command == "init":
        command_init(args)
    elif args.command == "run-round":
        command_run_round(args)
    elif args.command == "run-all":
        command_run_all(args)
    elif args.command == "status":
        command_status(args)


if __name__ == "__main__":
    main()
