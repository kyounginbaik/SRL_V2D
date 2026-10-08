"""Render initial and selected video-only public meshes for visual review."""
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark')
OUT = ROOT / 'visual_comparison'
PANEL_W, PANEL_H = 330, 260
MESH_FACES = 8000


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/{name}', size)


def canonical_vertices(vertices):
    centered = vertices - np.median(vertices, axis=0)
    covariance = np.cov(centered, rowvar=False)
    _, axes = np.linalg.eigh(covariance)
    coords = centered @ axes[:, ::-1]
    for i in range(3):
        if np.mean(coords[:, i] ** 3) < 0:
            coords[:, i] *= -1
    extent = np.max(np.ptp(coords, axis=0))
    return coords / max(extent, 1e-8)


def render_mesh(path, view, color):
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    mesh = mesh.simplify_quadric_decimation(face_count=min(MESH_FACES, len(mesh.faces)))
    verts = canonical_vertices(np.asarray(mesh.vertices))
    faces = np.asarray(mesh.faces)
    forward = np.asarray(view, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0., 0., 1.])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 0.1:
        right = np.cross(np.array([0., 1., 0.]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    xy = np.stack((verts @ right, verts @ up), axis=1)
    xy -= (xy.min(axis=0) + xy.max(axis=0)) / 2
    scale = 0.83 * min((PANEL_W - 24) / max(np.ptp(xy[:, 0]), 1e-8),
                       (PANEL_H - 24) / max(np.ptp(xy[:, 1]), 1e-8))
    pixels = xy * scale + np.array([PANEL_W / 2, PANEL_H / 2])
    pixels[:, 1] = PANEL_H - pixels[:, 1]
    triangles = verts[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.4, -0.3, 0.85]); light /= np.linalg.norm(light)
    shade = 0.40 + 0.60 * np.abs(normals @ light)
    depth = triangles.mean(axis=1) @ forward
    canvas = np.full((PANEL_H, PANEL_W, 3), (245, 248, 252), np.uint8)
    for index in np.argsort(depth):
        polygon = np.rint(pixels[faces[index]]).astype(np.int32)
        rgb = np.clip(np.asarray(color) * shade[index], 0, 255).astype(np.uint8)
        cv2.fillConvexPoly(canvas, polygon, tuple(int(x) for x in rgb[::-1]))
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def make_mesh_sheets():
    before = json.loads((ROOT / 'cd_o_summary.json').read_text())['objects']
    after = json.loads((ROOT / 'candidate_summary.json').read_text())['objects']
    names = list(after)
    OUT.mkdir(parents=True, exist_ok=True)
    views = ((1.0, 1.0, 0.8), (0.2, -0.3, 1.0))
    row_h, label_w, gap = 310, 215, 14
    width = label_w + 4 * PANEL_W + 5 * gap
    for page in range(2):
        subset = names[page * 5:(page + 1) * 5]
        sheet = Image.new('RGB', (width, 118 + row_h * len(subset)), '#ffffff')
        draw = ImageDraw.Draw(sheet)
        draw.text((26, 14), 'Public Track 3 | reconstructed meshes', font=font(28, True), fill='#172236')
        draw.text((26, 52), 'Initial SAM3D 0.6248 cm  →  selected video-only 0.3200 cm',
                  font=font(19), fill='#3d526b')
        draw.text((26, 82), 'Oblique and elevated views; each mesh independently oriented for display',
                  font=font(15), fill='#65748a')
        for row, name in enumerate(subset):
            y = 118 + row * row_h
            draw.rounded_rectangle((12, y, width - 12, y + row_h - 8),
                                   radius=12, fill='#f4f7fb')
            title = name.replace('_', ' ')
            draw.text((26, y + 20), title, font=font(19, True), fill='#172236')
            initial = float(before[name]['mean_cd_o_cm'])
            selected = float(after[name]['mean_cd_o_cm'])
            draw.text((26, y + 72), f'{initial:.3f} → {selected:.3f} cm',
                      font=font(16), fill='#52657c')
            draw.text((26, y + 105), 'improved' if selected < initial - 1e-4 else 'unchanged',
                      font=font(15, True), fill='#207e6c' if selected < initial - 1e-4 else '#6b7788')
            paths = (ROOT / 'meshes' / name / 'generated.glb', Path(after[name]['mesh']))
            for group, path in enumerate(paths):
                start = label_w + gap + group * (2 * PANEL_W + 2 * gap)
                draw.text((start + 8, y + 9), 'BEFORE' if group == 0 else 'AFTER',
                          font=font(16, True), fill='#416b9d' if group == 0 else '#208875')
                for view_idx, view in enumerate(views):
                    panel = render_mesh(path, view,
                                        (78, 121, 172) if group == 0 else (40, 158, 136))
                    sheet.paste(panel, (start + view_idx * (PANEL_W + gap), y + 36))
        target = OUT / f'mesh_before_after_{page + 1}.png'
        sheet.save(target)
        print(target, flush=True)


def make_mask_sheet():
    frame = Image.open('/data1/SRL_V2D/challenge_reconstruction/public/episode_000023/frames/000000.png').convert('RGB')
    crops = {
        'wooden_piece_1': (450, 730, 1050, 1450),
        'wooden_piece_2': (1030, 730, 1640, 1450),
    }
    result = Image.new('RGB', (1260, 1620), '#ffffff')
    draw = ImageDraw.Draw(result)
    draw.text((30, 18), 'Wooden pieces | mask correction', font=font(30, True), fill='#172236')
    draw.text((30, 62), 'Red: initial mask    Green: corrected mask', font=font(18), fill='#52657c')
    for row, (name, box) in enumerate(crops.items()):
        top = 120 + row * 740
        draw.text((26, top), name.replace('_', ' '), font=font(24, True), fill='#172236')
        for col, suffix in enumerate(('initial', 'corrected')):
            filename = 'mask_000000_initial.png' if col == 0 else 'mask_000000.png'
            mask = np.asarray(Image.open(ROOT / 'inputs' / name / filename).convert('L')) > 0
            pixels = np.asarray(frame).copy()
            color = np.array((240, 50, 50) if col == 0 else (20, 200, 90))
            pixels[mask] = (0.55 * pixels[mask] + 0.45 * color).astype(np.uint8)
            crop = Image.fromarray(pixels).crop(box).resize((590, 640))
            result.paste(crop, (25 + col * 620, top + 42))
            draw.text((25 + col * 620, top + 689), 'BEFORE' if col == 0 else 'AFTER',
                      font=font(17, True), fill='#b23e44' if col == 0 else '#208875')
    target = OUT / 'wooden_piece_masks_before_after.png'
    result.save(target)
    print(target, flush=True)


if __name__ == '__main__':
    make_mesh_sheets()
    make_mask_sheet()
