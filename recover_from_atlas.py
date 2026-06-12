import argparse
import json
import os
import re

import cv2
import numpy as np


def _resolve_meta_path(meta_arg: str) -> str:
    """Resolve meta/record path, searching under visual_examples by default."""
    raw = str(meta_arg or "").strip()
    if len(raw) == 0:
        raw = "record103.js"

    # 1) explicit path
    if os.path.isfile(raw):
        return raw

    # 2) likely locations under workspace
    base_candidates = [
        raw,
        os.path.join("visual_examples", raw),
        os.path.join("visual_examples", "record", raw),
    ]

    root, ext = os.path.splitext(raw)
    if ext == "":
        base_candidates.extend(
            [
                f"{raw}.js",
                f"{raw}.json",
                os.path.join("visual_examples", f"{raw}.js"),
                os.path.join("visual_examples", f"{raw}.json"),
                os.path.join("visual_examples", "record", f"{raw}.js"),
                os.path.join("visual_examples", "record", f"{raw}.json"),
            ]
        )
    else:
        base_candidates.extend(
            [
                os.path.join("visual_examples", root + ext),
                os.path.join("visual_examples", "record", root + ext),
            ]
        )

    for p in base_candidates:
        p_norm = os.path.normpath(p)
        if os.path.isfile(p_norm):
            return p_norm

    # 3) recursive fallback in visual_examples
    visual_root = os.path.normpath("visual_examples")
    if os.path.isdir(visual_root):
        names = [raw]
        if ext == "":
            names.extend([f"{raw}.js", f"{raw}.json"])
        matches = []
        for dirpath, _, filenames in os.walk(visual_root):
            for n in names:
                if n in filenames:
                    matches.append(os.path.join(dirpath, n))
        if len(matches) == 1:
            return os.path.normpath(matches[0])
        if len(matches) > 1:
            print("Warning: multiple meta candidates found under visual_examples; using first:")
            print(f"  {matches[0]}")
            return os.path.normpath(matches[0])

    return os.path.normpath(base_candidates[-1])


def _load_meta(meta_path: str) -> dict:
    """Load metadata from either JSON meta or recordXX.js."""
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    text = ""
    with open(meta_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Fast path for pure JSON files.
    if str(meta_path).lower().endswith(".json"):
        return json.loads(text)

    # Try parse full content as JSON first.
    try:
        return json.loads(text)
    except Exception:
        pass

    # recordXX.js pattern: const meta = {...}; export default meta;
    m = re.search(r"const\s+meta\s*=\s*(\{[\s\S]*?\})\s*;\s*export\s+default\s+meta", text)
    if m is not None:
        obj_text = m.group(1)
    else:
        # fallback: first {...} span
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise ValueError(f"Unable to parse meta object from: {meta_path}")
        obj_text = text[start : end + 1]

    # Remove JS comments and trailing commas to improve robustness.
    obj_text = re.sub(r"/\*[\s\S]*?\*/", "", obj_text)
    obj_text = re.sub(r"//.*$", "", obj_text, flags=re.MULTILINE)
    obj_text = re.sub(r",\s*([}\]])", r"\1", obj_text)
    return json.loads(obj_text)


def _resolve_path_near_meta(meta_path: str, rel_or_abs: str) -> str:
    if not rel_or_abs:
        return ""
    candidate = str(rel_or_abs).replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(candidate) and os.path.exists(candidate):
        return candidate

    # 1) relative to meta dir
    p1 = os.path.normpath(os.path.join(os.path.dirname(meta_path), candidate))
    if os.path.exists(p1):
        return p1

    # 2) relative to parent of meta dir (for visual_examples/record/*.js -> visual_examples/*)
    p2 = os.path.normpath(os.path.join(os.path.dirname(meta_path), "..", candidate))
    if os.path.exists(p2):
        return p2

    # Return best-effort path for diagnostics.
    return p1


def _infer_secret_basename(meta_path: str, meta: dict) -> str:
    """Infer atlas filename stem, e.g., secret103 from meta/record file."""
    out_secret = meta.get("outputs", {}).get("secret", "")
    if isinstance(out_secret, str) and len(out_secret.strip()) > 0:
        stem = os.path.splitext(os.path.basename(out_secret.replace("\\", os.sep).replace("/", os.sep)))[0]
        if len(stem) > 0:
            return stem

    # Fallback: parse recordNNN from meta filename.
    name = os.path.basename(meta_path)
    m = re.search(r"record(\d+)", name, flags=re.IGNORECASE)
    if m is not None:
        return f"secret{m.group(1)}"

    return "secret"


def _resolve_atlas_path(meta_path: str, meta: dict, atlas_arg: str) -> str:
    """Resolve decoded atlas path with priority on main_results."""
    if isinstance(atlas_arg, str) and len(atlas_arg.strip()) > 0:
        atlas_manual = atlas_arg.strip()
        if os.path.isabs(atlas_manual):
            return atlas_manual
        return os.path.normpath(atlas_manual)

    secret_stem = _infer_secret_basename(meta_path, meta)
    atlas_name = f"{secret_stem}.png"

    candidates = [
        os.path.join("results", "fdns_scene_restoration", "secret", "main", "secret_rev_jpegloss", atlas_name),
        os.path.join("results", "fdns_scene_restoration", "secret", "secret_rev_jpegloss", atlas_name),
    ]

    for p in candidates:
        p_norm = os.path.normpath(p)
        if os.path.isfile(p_norm):
            return p_norm

    # Return primary expected location for diagnostics.
    return os.path.normpath(candidates[0])


def _extract_patch_from_atlas(atlas_img: np.ndarray, obj: dict) -> np.ndarray:
    """Extract object patch from atlas for both old-meta and record-meta schemas."""
    if "target" in obj:
        target = obj.get("target", {})
        source = obj.get("source", {})

        # New record schema.
        cx1, cy1, cx2, cy2 = [int(v) for v in target.get("cell_xyxy", [0, 0, 0, 0])]
        cell = atlas_img[cy1:cy2, cx1:cx2]
        if cell.size == 0:
            raise ValueError("Invalid target.cell_xyxy for current atlas shape")

        placed = target.get("placed_content_xyxy", None)
        if placed is not None and len(placed) == 4:
            px1, py1, px2, py2 = [int(v) for v in placed]

            # placed_content_xyxy may be global atlas coords or local cell coords.
            if px1 >= cx1 and py1 >= cy1 and px2 <= cx2 and py2 <= cy2:
                gx1, gy1, gx2, gy2 = px1, py1, px2, py2
            else:
                gx1, gy1, gx2, gy2 = cx1 + px1, cy1 + py1, cx1 + px2, cy1 + py2

            gx1 = max(0, min(gx1, atlas_img.shape[1]))
            gx2 = max(0, min(gx2, atlas_img.shape[1]))
            gy1 = max(0, min(gy1, atlas_img.shape[0]))
            gy2 = max(0, min(gy2, atlas_img.shape[0]))
            cropped = atlas_img[gy1:gy2, gx1:gx2]
        else:
            # fallback to letterbox removal from cell
            lb = target.get("letterbox", {})
            pad_left = int(lb.get("pad_left", 0))
            pad_top = int(lb.get("pad_top", 0))
            new_w = int(lb.get("new_w", cell.shape[1]))
            new_h = int(lb.get("new_h", cell.shape[0]))
            cropped = cell[pad_top : pad_top + new_h, pad_left : pad_left + new_w]

        if cropped.size == 0:
            raise ValueError("Empty crop extracted from atlas")

        # Resize back to the source patch size used before packing.
        src_wh = target.get("src_patch_size_wh", None)
        if (not src_wh) and isinstance(source, dict):
            src_wh = source.get("extract", {}).get("patch_size_wh", None)

        if src_wh and len(src_wh) == 2:
            sw, sh = int(src_wh[0]), int(src_wh[1])
            if sw > 0 and sh > 0:
                cropped = cv2.resize(cropped, (sw, sh), interpolation=cv2.INTER_CUBIC)

        return cropped

    # Old schema from test_meta.json.
    cx1, cy1, cx2, cy2 = [int(v) for v in obj["cell_xyxy"]]
    cell = atlas_img[cy1:cy2, cx1:cx2]
    lb = obj.get("letterbox", {})
    pad_top = int(lb.get("pad_top", 0))
    pad_left = int(lb.get("pad_left", 0))
    new_h = int(lb.get("new_h", cell.shape[0]))
    new_w = int(lb.get("new_w", cell.shape[1]))
    obj_cropped = cell[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    bx1, by1, bx2, by2 = [int(v) for v in obj["bbox_xyxy"]]
    orig_w = max(1, bx2 - bx1)
    orig_h = max(1, by2 - by1)
    return cv2.resize(obj_cropped, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)


def _build_patch_mask(obj: dict, patch_h: int, patch_w: int) -> np.ndarray:
    """Build object-valid mask in patch coordinates."""
    mask = np.zeros((patch_h, patch_w), dtype=np.uint8)

    poly = obj.get("source", {}).get("extract", {}).get("poly_patch_xy", None)
    if poly is None:
        mask.fill(255)
        return mask

    pts = None
    if isinstance(poly, list) and len(poly) > 0 and isinstance(poly[0], list):
        pts = np.array(poly, dtype=np.float32)
    elif isinstance(poly, list) and len(poly) % 2 == 0:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)

    if pts is None or pts.shape[0] < 3:
        mask.fill(255)
        return mask

    pts[:, 0] = np.clip(pts[:, 0], 0, patch_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, patch_h - 1)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


def _shrink_mask(mask: np.ndarray, shrink: int) -> np.ndarray:
    if shrink <= 0:
        return mask
    k = int(2 * shrink + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, kernel, iterations=1)


def _blend_full_frame(
    full_scene: np.ndarray,
    warped_patch: np.ndarray,
    warped_mask: np.ndarray,
    mode: str,
    feather_ksize: int,
) -> np.ndarray:
    """Blend a warped object into scene using a full-frame mask."""
    out = full_scene.copy()
    valid = warped_mask > 0
    if not np.any(valid):
        return out

    if mode == "direct":
        out[valid] = warped_patch[valid]
        return out

    if mode == "seamless":
        ys, xs = np.where(valid)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1

        src_roi = warped_patch[y1:y2, x1:x2]
        mask_roi = warped_mask[y1:y2, x1:x2]
        center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
        try:
            out = cv2.seamlessClone(src_roi, out, mask_roi, center, cv2.NORMAL_CLONE)
        except Exception as e:
            print(f"  - seamlessClone failed, fallback to direct: {e}")
            out[valid] = warped_patch[valid]
        return out

    # feather mode
    k = max(3, int(feather_ksize))
    if k % 2 == 0:
        k += 1

    alpha = warped_mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha3 = np.repeat(alpha[:, :, None], 3, axis=2)

    out_f = out.astype(np.float32)
    src_f = warped_patch.astype(np.float32)
    out = (src_f * alpha3 + out_f * (1.0 - alpha3)).astype(np.uint8)
    return out


def recover_objects():
    parser = argparse.ArgumentParser(
        description="Recover objects from decoded atlas and paste back using record meta (json/js)."
    )
    parser.add_argument(
        "--meta",
        default="record103.js",
        help="Metadata file path or filename. If filename, it is auto-searched under visual_examples/.",
    )
    parser.add_argument(
        "--atlas",
        default="",
        help="Optional decoded atlas path. If empty, auto-resolve from main_results using meta.",
    )
    parser.add_argument(
        "--cover",
        default="",
        help="Optional cover path. If empty, auto-resolve from meta.outputs.cover",
    )
    parser.add_argument(
        "--output",
        default="results/recovered_full_scene.png",
        help="Path to save recovered full scene",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "seamless", "feather"],
        default="feather",
        help="Blending mode",
    )
    parser.add_argument(
        "--shrink",
        type=int,
        default=2,
        help="Erode warped object mask by N pixels before blending",
    )
    parser.add_argument(
        "--feather-ksize",
        type=int,
        default=11,
        help="Gaussian kernel size for feather blending (odd number recommended)",
    )
    args = parser.parse_args()

    meta_path = _resolve_meta_path(args.meta)
    try:
        meta = _load_meta(meta_path)
    except Exception as e:
        print(f"Error loading meta from '{meta_path}': {e}")
        return

    atlas_path = _resolve_atlas_path(args.meta, meta, args.atlas)
    if not os.path.isfile(atlas_path):
        print(
            "Error: Atlas image not found. "
            f"Requested='{args.atlas}', resolved='{atlas_path}'."
        )
        return

    if args.cover:
        cover_path = args.cover
    else:
        cover_rel = meta.get("outputs", {}).get("cover", "")
        cover_path = _resolve_path_near_meta(meta_path, cover_rel)

    if not os.path.isfile(cover_path):
        print(
            "Error: Cover image not found. "
            f"Requested='{args.cover}', resolved='{cover_path}'. "
            "You can pass --cover explicitly."
        )
        return

    atlas_img = cv2.imread(atlas_path, cv2.IMREAD_COLOR)
    full_scene = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    if atlas_img is None:
        print(f"Error: Failed to read atlas image: {atlas_path}")
        return
    if full_scene is None:
        print(f"Error: Failed to read cover image: {cover_path}")
        return

    print(f"Loaded atlas: {atlas_path} shape={atlas_img.shape}")
    print(f"Loaded cover: {cover_path} shape={full_scene.shape}")

    objects = meta.get("objects", [])
    if not isinstance(objects, list) or len(objects) == 0:
        print("Error: meta.objects is empty")
        return

    print(f"Found {len(objects)} objects to recover")

    for i, obj in enumerate(objects):
        label = str(obj.get("label", f"obj_{i}"))
        print(f"[{i}] Recovering {label} ...")

        try:
            patch = _extract_patch_from_atlas(atlas_img, obj)
        except Exception as e:
            print(f"  - skip: failed to extract patch: {e}")
            continue

        # Preferred path for record schema: project patch back with homography.
        h_patch_to_src = obj.get("source", {}).get("extract", {}).get("H_patch_to_src", None)
        if h_patch_to_src is not None and len(h_patch_to_src) == 9:
            H = np.array(h_patch_to_src, dtype=np.float32).reshape(3, 3)

            mask_patch = _build_patch_mask(obj, patch.shape[0], patch.shape[1])
            mask_patch = _shrink_mask(mask_patch, int(args.shrink))

            warped_patch = cv2.warpPerspective(
                patch,
                H,
                (full_scene.shape[1], full_scene.shape[0]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            warped_mask = cv2.warpPerspective(
                mask_patch,
                H,
                (full_scene.shape[1], full_scene.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            full_scene = _blend_full_frame(
                full_scene,
                warped_patch,
                warped_mask,
                mode=args.mode,
                feather_ksize=int(args.feather_ksize),
            )
            print("  - restored by homography projection")
            continue

        # Fallback path for old meta schema: bbox paste.
        bbox = obj.get("bbox_xyxy", None)
        if bbox is None:
            bbox = obj.get("source", {}).get("bbox_xyxy", None)

        if bbox is None or len(bbox) != 4:
            print("  - skip: neither H_patch_to_src nor bbox available")
            continue

        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        bx1 = max(0, min(bx1, full_scene.shape[1]))
        bx2 = max(0, min(bx2, full_scene.shape[1]))
        by1 = max(0, min(by1, full_scene.shape[0]))
        by2 = max(0, min(by2, full_scene.shape[0]))

        if bx2 <= bx1 or by2 <= by1:
            print("  - skip: invalid bbox")
            continue

        resized = cv2.resize(patch, (bx2 - bx1, by2 - by1), interpolation=cv2.INTER_CUBIC)
        roi_bg = full_scene[by1:by2, bx1:bx2]

        roi_mask = np.full((by2 - by1, bx2 - bx1), 255, dtype=np.uint8)
        roi_mask = _shrink_mask(roi_mask, int(args.shrink))

        if args.mode == "direct":
            roi_bg[roi_mask > 0] = resized[roi_mask > 0]
        elif args.mode == "seamless":
            center = (bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2)
            try:
                full_scene = cv2.seamlessClone(resized, full_scene, roi_mask, center, cv2.NORMAL_CLONE)
            except Exception as e:
                print(f"  - seamlessClone failed, fallback to direct: {e}")
                roi_bg[roi_mask > 0] = resized[roi_mask > 0]
        else:
            k = max(3, int(args.feather_ksize))
            if k % 2 == 0:
                k += 1
            alpha = cv2.GaussianBlur(roi_mask.astype(np.float32) / 255.0, (k, k), 0)
            alpha3 = np.repeat(alpha[:, :, None], 3, axis=2)
            mix = resized.astype(np.float32) * alpha3 + roi_bg.astype(np.float32) * (1.0 - alpha3)
            roi_bg[:, :, :] = mix.astype(np.uint8)

        full_scene[by1:by2, bx1:bx2] = roi_bg
        print("  - restored by bbox fallback")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ok = cv2.imwrite(args.output, full_scene)
    if not ok:
        print(f"Error: failed to save output: {args.output}")
        return
    print(f"Success! Saved recovered scene to: {args.output}")


if __name__ == "__main__":
    recover_objects()
