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
    parser.add_argument("--workflow_config", "--config_file", dest="workflow_config", help="Workflow JSON with defaults and rounds.")
    parser.add_argument("--target_ids", nargs="+", type=int, help="Legacy shorthand: object ids to process as one-id rounds.")
    parser.add_argument(
        "--target_groups",
        nargs="+",
        help="Round groups such as '34,57 81 90,91'. Each token is one inpaint round.",
    )
    parser.add_argument("--round_index", type=int, default=0, help="Round index for run-round.")
    parser.add_argument("--base_model_path", help="Optional trained 3DGIC model path. Defaults to --model_path.")
    parser.add_argument("--removal_config_template")
    parser.add_argument("--inpaint_config_template")
    parser.add_argument("--top_k_ref_views", type=int)
    parser.add_argument("--simple_lama_device")
    parser.add_argument("--mask_threshold", type=int)
    parser.add_argument("--mask_dilation", type=int)
    parser.add_argument("--intersect_top_m", type=int)
    parser.add_argument("--intersect_cache_size", type=int)
    parser.add_argument("--background_id", type=int)
    parser.add_argument("--finetune_iteration", type=int, default=None)
    parser.add_argument("--type", choices=("render", "neilf"), default="render")
    parser.add_argument("--render_intermediate", action="store_true", help="Render inpaint outputs for non-final rounds.")
    parser.add_argument(
        "--storage_mode",
        choices=("full", "lite", "minimal"),
        default="full",
        help="Output retention mode. minimal keeps scene_out plus final render snapshots and removes heavy round intermediates.",
    )
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


def normalize_id_list(value, field_name="target_id"):
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                return normalize_id_list(json.loads(text), field_name=field_name)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON list for {field_name}: {value}") from exc
        for separator in (",", "+", "_"):
            text = text.replace(separator, " ")
        return [int(item) for item in text.split()]
    if isinstance(value, (list, tuple)):
        ids = []
        for item in value:
            ids.extend(normalize_id_list(item, field_name=field_name))
        return [int(item) for item in ids]
    raise TypeError(f"{field_name} must be int, str, or list, got {type(value).__name__}")


def normalize_round_spec(spec, round_index):
    if isinstance(spec, dict):
        normalized = dict(spec)
        raw_target = normalized.pop("target_ids", normalized.get("target_id", None))
    else:
        normalized = {}
        raw_target = spec
    target_ids = normalize_id_list(raw_target, field_name=f"rounds[{round_index}].target_id")
    if not target_ids:
        raise ValueError(f"Round {round_index} has empty target_id/target_ids")
    normalized["target_ids"] = target_ids
    normalized["target_id"] = target_ids
    normalized["target_tag"] = target_tag(target_ids)
    return normalized


def target_tag(target_ids):
    return "_".join(str(item) for item in normalize_id_list(target_ids))


def parse_target_groups(groups):
    if not groups:
        return []
    return [{"target_ids": normalize_id_list(group, field_name="target_groups")} for group in groups]


def load_workflow_config(path):
    path = Path(path).expanduser().resolve()
    workflow = read_json(path)
    workflow.setdefault("defaults", {})
    rounds = workflow.get("rounds")
    if not rounds:
        raise ValueError(f"'rounds' must be a non-empty list in workflow config: {path}")
    workflow["rounds"] = [normalize_round_spec(spec, idx) for idx, spec in enumerate(rounds)]
    workflow["_config_path"] = str(path)
    workflow["_config_dir"] = str(path.parent)
    return workflow


def resolve_config_path(value, config_dir=None):
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute() or config_dir is None:
        return str(path.resolve())
    candidate = Path(config_dir) / path
    if candidate.exists():
        return str(candidate.resolve())
    return str(path.resolve())


def workflow_rounds(workflow):
    if "rounds" in workflow:
        return [normalize_round_spec(spec, idx) for idx, spec in enumerate(workflow["rounds"])]
    return [normalize_round_spec(item, idx) for idx, item in enumerate(workflow.get("target_ids", []))]


def completed_ids_before(rounds, round_index):
    completed = []
    for spec in rounds[:round_index]:
        completed.extend(spec["target_ids"])
    return completed


def workflow_option(workflow, round_spec, key, fallback=None):
    if key in round_spec:
        return round_spec[key]
    defaults = workflow.get("defaults", {})
    if key in defaults:
        return defaults[key]
    if key in workflow:
        return workflow[key]
    return fallback


def is_last_round(workflow, round_index):
    return round_index == len(workflow_rounds(workflow)) - 1


def should_render_inpaint(args, workflow, round_index):
    if args.storage_mode == "full":
        return True
    return args.render_intermediate or is_last_round(workflow, round_index)


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


def remove_path(path):
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


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


def find_image_by_stem(folder, stem):
    folder = Path(folder)
    for ext in sorted(IMAGE_EXTENSIONS):
        candidate = folder / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    if folder.is_dir():
        matches = sorted(item for item in folder.iterdir() if item.is_file() and item.stem == stem)
        if matches:
            return matches[0]
    return None


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

    # With intersect_top_m enabled, only the selected views have expensive
    # intersect masks. The finetune loader still expects every training view to
    # have a 2D mask, so fall back to the ordinary removal mask for the rest.
    fallback_mask_dir = removal_output / "remove_objects_pred"
    filled = 0
    missing = 0
    for image_path in sorted((inpaint_root / "images").iterdir()):
        if not image_path.is_file() or image_path.suffix not in IMAGE_EXTENSIONS:
            continue
        if find_image_by_stem(inpaint_root, image_path.stem) is not None:
            continue
        fallback = find_image_by_stem(fallback_mask_dir, image_path.stem)
        if fallback is None:
            missing += 1
            continue
        shutil.copy2(fallback, inpaint_root / f"{image_path.stem}{fallback.suffix}")
        filled += 1
    if filled or missing:
        print(f"Filled missing inpaint masks from remove_objects_pred: filled={filled}, missing={missing}")
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


def load_intersect_candidate_stems(removal_output):
    path = Path(removal_output) / "intersect_candidates.json"
    if not path.is_file():
        return None
    data = read_json(path)
    return {str(stem) for stem in data.get("stems", [])}


def select_reference_views(inpaint_root, removal_output, top_k, mask_threshold):
    inpaint_root = Path(inpaint_root)
    candidate_stems = load_intersect_candidate_stems(removal_output)
    areas = []
    for mask_path in sorted(inpaint_root.glob("*.png")):
        if candidate_stems is not None and mask_path.stem not in candidate_stems:
            continue
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


def write_round_configs(workflow, paths, round_spec, refs, round_index):
    target_ids = [int(item) for item in round_spec["target_ids"]]
    removal_config = load_template(workflow["removal_config_template"])
    removal_config.update({
        "select_obj_id": target_ids,
        "images": "images",
        "object_path": "object_mask",
        "source_path": str(paths["data_work"]),
        "intersect_top_m": int(workflow_option(workflow, round_spec, "intersect_top_m")),
        "intersect_cache_size": int(workflow_option(workflow, round_spec, "intersect_cache_size")),
    })
    removal_overrides = workflow_option(workflow, round_spec, "removal_config_overrides", {})
    if removal_overrides:
        removal_config.update(removal_overrides)
    write_json(paths["removal_config"], removal_config)

    exp_setting = f"iter3dgic_round_{round_index:03d}_obj_{round_spec['target_tag']}"
    inpaint_config = load_template(workflow["inpaint_config_template"])
    inpaint_config.update({
        "select_obj_id": target_ids,
        "images": "substitude",
        "object_path": "inpaint_2d_unseen_mask",
        "source_path": str(paths["data_work"]),
        "use_ref": 1,
        "ref_names": [item["ref_name"] for item in refs],
        "ref_id": [item["ref_id"] for item in refs if item.get("ref_id") is not None],
        "finetune_iteration": int(workflow_option(workflow, round_spec, "finetune_iteration")),
        "exp_setting": exp_setting,
    })
    inpaint_overrides = workflow_option(workflow, round_spec, "inpaint_config_overrides", {})
    if inpaint_overrides:
        inpaint_config.update(inpaint_overrides)
    write_json(paths["inpaint_config"], inpaint_config)
    return exp_setting


def copy_substitude_images(data_work):
    inpaint_images = Path(data_work) / "inpaint_2d_unseen_mask" / "images"
    substitude = Path(data_work) / "substitude"
    copy_dir_contents(inpaint_images, substitude)


def run_simple_lama(workflow, round_spec, inpaint_root, refs):
    stems = [item["stem"] for item in refs]
    cmd = [
        sys.executable,
        "tools/simple_lama_3dgic.py",
        "--inpaint_dir",
        inpaint_root,
        "--stems",
        *stems,
        "--device",
        workflow_option(workflow, round_spec, "simple_lama_device"),
        "--mask_dilation",
        workflow_option(workflow, round_spec, "mask_dilation"),
        "--mask_threshold",
        workflow_option(workflow, round_spec, "mask_threshold"),
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


def inpaint_render_root(exp_setting, finetune_iteration):
    final_iteration = int(finetune_iteration) - 1
    return f"ours_object_inpaint/{exp_setting}_iteration_{final_iteration}"


def save_final_render_snapshot(paths, exp_setting, finetune_iteration):
    copied = []
    rel_root = inpaint_render_root(exp_setting, finetune_iteration)
    snapshot_root = paths["scene_out"] / "final_render"
    for split in ("train", "test"):
        src = paths["scene_in"] / split / rel_root
        if not src.is_dir():
            continue
        dst = snapshot_root / split
        if dst.exists() or dst.is_symlink():
            remove_path(dst)
        shutil.copytree(src, dst)
        copied.append(str(dst))
    return copied


def cleanup_lite_round(paths):
    removed = []
    for path in [
        paths["scene_in"] / "point_cloud_object_removal",
        paths["scene_in"] / "point_cloud_object_inpaint",
    ]:
        if path.exists() or path.is_symlink():
            remove_path(path)
            removed.append(str(path))
    return removed


def cleanup_minimal_round(paths):
    removed = []
    for path in [paths["data_work"], paths["scene_in"]]:
        if path.exists() or path.is_symlink():
            remove_path(path)
            removed.append(str(path))
    return removed


def command_init(args):
    if args.type != "render":
        raise ValueError("iterative_3dgic v1 only supports --type render")
    if not args.source_path:
        raise ValueError("--source_path is required for init")

    config_workflow = load_workflow_config(args.workflow_config) if args.workflow_config else {}
    config_dir = config_workflow.get("_config_dir")
    if args.target_groups:
        raw_rounds = parse_target_groups(args.target_groups)
    elif args.target_ids:
        raw_rounds = [{"target_ids": [int(item)]} for item in args.target_ids]
    elif config_workflow:
        raw_rounds = config_workflow["rounds"]
    else:
        raise ValueError("--workflow_config/--config_file or --target_groups/--target_ids is required for init")
    rounds = [normalize_round_spec(spec, idx) for idx, spec in enumerate(raw_rounds)]

    root = workflow_root(args.model_path)
    if root.exists() and not args.overwrite:
        raise FileExistsError(f"Workflow already exists. Use --overwrite to replace: {root}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    defaults = dict(config_workflow.get("defaults", {}))
    removal_config_template = (
        args.removal_config_template
        or defaults.get("removal_config_template")
        or config_workflow.get("removal_config_template")
        or str(PROJECT_ROOT / "configs/object_removal/bear.json")
    )
    inpaint_config_template = (
        args.inpaint_config_template
        or defaults.get("inpaint_config_template")
        or config_workflow.get("inpaint_config_template")
        or str(PROJECT_ROOT / "configs/object_inpaint/bear_new.json")
    )
    removal_config_template = resolve_config_path(removal_config_template, config_dir)
    inpaint_config_template = resolve_config_path(inpaint_config_template, config_dir)
    inpaint_template = load_template(inpaint_config_template)
    finetune_iteration = (
        args.finetune_iteration
        or defaults.get("finetune_iteration")
        or config_workflow.get("finetune_iteration")
        or int(inpaint_template.get("finetune_iteration", 10000))
    )
    workflow = {
        "base_model_path": str(Path(args.base_model_path or args.model_path).expanduser().resolve()),
        "source_path": str(Path(args.source_path).expanduser().resolve()),
        "workflow_config": str(Path(args.workflow_config).expanduser().resolve()) if args.workflow_config else None,
        "rounds": rounds,
        "target_groups": [spec["target_ids"] for spec in rounds],
        "target_ids": [spec["target_ids"][0] for spec in rounds if len(spec["target_ids"]) == 1],
        "defaults": defaults,
        "top_k_ref_views": int(args.top_k_ref_views or defaults.get("top_k_ref_views", config_workflow.get("top_k_ref_views", 3))),
        "simple_lama_device": args.simple_lama_device or defaults.get("simple_lama_device", config_workflow.get("simple_lama_device", "cuda")),
        "mask_threshold": int(args.mask_threshold if args.mask_threshold is not None else defaults.get("mask_threshold", config_workflow.get("mask_threshold", 0))),
        "mask_dilation": int(args.mask_dilation if args.mask_dilation is not None else defaults.get("mask_dilation", config_workflow.get("mask_dilation", 0))),
        "intersect_top_m": int(args.intersect_top_m if args.intersect_top_m is not None else defaults.get("intersect_top_m", config_workflow.get("intersect_top_m", 20))),
        "intersect_cache_size": int(args.intersect_cache_size if args.intersect_cache_size is not None else defaults.get("intersect_cache_size", config_workflow.get("intersect_cache_size", 256))),
        "background_id": int(args.background_id if args.background_id is not None else defaults.get("background_id", config_workflow.get("background_id", 0))),
        "removal_config_template": removal_config_template,
        "inpaint_config_template": inpaint_config_template,
        "finetune_iteration": int(finetune_iteration),
        "type": args.type,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(root / "workflow.json", workflow)
    for round_index, spec in enumerate(workflow["rounds"]):
        paths = round_paths(root, round_index)
        paths["round_dir"].mkdir(parents=True, exist_ok=True)
        save_meta(
            paths,
            status="initialized",
            round_index=round_index,
            target_ids=spec["target_ids"],
            target_tag=spec["target_tag"],
            completed_before=completed_ids_before(workflow["rounds"], round_index),
        )
    print(f"Initialized workflow: {root / 'workflow.json'}")


def command_run_round(args, workflow=None, round_index=None):
    workflow = workflow or load_workflow(args.model_path)
    if workflow.get("type", "render") != "render":
        raise ValueError("iterative_3dgic v1 only supports render workflows")

    root = workflow_root(args.model_path)
    round_index = args.round_index if round_index is None else round_index
    rounds = workflow_rounds(workflow)
    if round_index < 0 or round_index >= len(rounds):
        raise IndexError(f"round_index={round_index} is outside rounds range")

    round_spec = rounds[round_index]
    target_ids = [int(item) for item in round_spec["target_ids"]]
    paths = round_paths(root, round_index)
    paths["round_dir"].mkdir(parents=True, exist_ok=True)
    completed_before = completed_ids_before(rounds, round_index)
    save_meta(
        paths,
        status="running",
        round_index=round_index,
        target_ids=target_ids,
        target_tag=round_spec["target_tag"],
        completed_before=completed_before,
        storage_mode=args.storage_mode,
        render_intermediate=bool(args.render_intermediate),
    )

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
        "select_obj_id": target_ids,
        "images": "images",
        "object_path": "object_mask",
        "source_path": str(paths["data_work"]),
        "intersect_top_m": int(workflow_option(workflow, round_spec, "intersect_top_m")),
        "intersect_cache_size": int(workflow_option(workflow, round_spec, "intersect_cache_size")),
    })
    removal_overrides = workflow_option(workflow, round_spec, "removal_config_overrides", {})
    if removal_overrides:
        removal_config.update(removal_overrides)
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
        int(workflow_option(workflow, round_spec, "top_k_ref_views")),
        int(workflow_option(workflow, round_spec, "mask_threshold")),
    )
    exp_setting = write_round_configs(workflow, paths, round_spec, refs, round_index)
    save_meta(paths, status="refs_selected", selected_refs=refs, exp_setting=exp_setting)

    try:
        run_simple_lama(workflow, round_spec, inpaint_root, refs)
    except Exception:
        save_meta(paths, status="lama_failed")
        raise
    save_meta(paths, status="lama_outputs_ready")

    copy_substitude_images(paths["data_work"])
    render_inpaint = should_render_inpaint(args, workflow, round_index)
    inpaint_cmd = [
        sys.executable,
        "edit_object_inpaint_spin.py",
        "-m",
        paths["scene_in"],
        "--config_file",
        paths["inpaint_config"],
        "--skip_test",
    ]
    if not render_inpaint:
        inpaint_cmd.append("--skip_train")
    run_command(inpaint_cmd)
    save_meta(paths, status="inpaint_finished")

    promote_round_output(paths, exp_setting, workflow_option(workflow, round_spec, "finetune_iteration"), args.overwrite)
    final_render_snapshot = []
    if args.storage_mode == "minimal" and render_inpaint:
        final_render_snapshot = save_final_render_snapshot(
            paths,
            exp_setting,
            workflow_option(workflow, round_spec, "finetune_iteration"),
        )

    storage_cleanup = []
    if args.storage_mode == "minimal":
        storage_cleanup = cleanup_minimal_round(paths)
    elif args.storage_mode == "lite":
        storage_cleanup = cleanup_lite_round(paths)

    save_meta(
        paths,
        status="completed",
        scene_out=str(paths["scene_out"]),
        rendered_inpaint=bool(render_inpaint),
        final_render_snapshot=final_render_snapshot,
        storage_cleanup=storage_cleanup,
    )


def command_run_all(args):
    workflow = load_workflow(args.model_path)
    for round_index in range(len(workflow_rounds(workflow))):
        command_run_round(args, workflow=workflow, round_index=round_index)


def command_status(args):
    workflow = load_workflow(args.model_path)
    root = workflow_root(args.model_path)
    rounds = workflow_rounds(workflow)
    print(f"workflow: {root / 'workflow.json'}")
    print(f"target_groups: {[spec['target_ids'] for spec in rounds]}")
    for round_index, spec in enumerate(rounds):
        meta_path = round_paths(root, round_index)["meta"]
        if meta_path.is_file():
            meta = read_json(meta_path)
            status = meta.get("status", "unknown")
            selected = meta.get("selected_refs", [])
            refs = ",".join(str(item.get("stem")) for item in selected)
            storage_mode = meta.get("storage_mode", "")
            rendered = meta.get("rendered_inpaint", "")
        else:
            status = "missing"
            refs = ""
            storage_mode = ""
            rendered = ""
        details = []
        if storage_mode:
            details.append(f"storage={storage_mode}")
        if rendered != "":
            details.append(f"rendered={rendered}")
        detail_text = (" " + " ".join(details)) if details else ""
        print(f"round_{round_index:03d} target={spec['target_tag']} status={status} refs={refs}{detail_text}")


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
