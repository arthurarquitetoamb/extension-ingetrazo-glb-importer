# SPDX-License-Identifier: GPL-3.0-or-later
"""Import glTF/GLB files while keeping their node hierarchy as IngeTrazo groups."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox

from core.group import Group
from core.history import Command, SnapshotImport
from core.materials import Material
from core.mesh import Mesh

log = logging.getLogger("ingetrazo.plugins.gltf_group_importer")


def _material_for_gltf(glb, js: dict, material_index: int | None, images):
    """Return a glTF material recipe and its IngeTrazo material identity."""
    texture_path, uv_transform, rgba, uv_set, opacity = glb._material_recipe(
        js, material_index, images
    )
    if rgba is not None:
        rgba = [float(channel) for channel in rgba]

    color = tuple(rgba[:3]) if rgba else None
    finish = None
    raw = js.get("materials", [])[material_index] if material_index is not None else {}
    default_name = (
        f"Material {material_index + 1}"
        if material_index is not None
        else "Material"
    )
    name = str(raw.get("name") or default_name)
    finish = (raw.get("extras") or {}).get("ingetrazo_finish")
    from core.finish import PBR, resolve

    if finish not in PBR:
        pbr = raw.get("pbrMetallicRoughness") or {}
        transmission = (
            raw.get("extensions", {})
            .get("KHR_materials_transmission", {})
            .get("transmissionFactor", 0.0)
        )
        ior = raw.get("extensions", {}).get("KHR_materials_ior", {}).get("ior", 1.5)
        target = (
            float(pbr.get("roughnessFactor", 1.0)),
            float(pbr.get("metallicFactor", 1.0)),
            float(transmission),
            float(ior),
        )
        finish = min(
            PBR,
            key=lambda key: sum(
                (actual - expected) ** 2
                for actual, expected in zip(target, PBR[key])
            ),
        ) if raw else resolve(None, name, texture_path, opacity)

    texture = (
        {"path": texture_path, "sw": 1.0, "sh": 1.0}
        if texture_path
        else None
    )
    material = Material(
        name=name,
        color=color,
        texture=texture,
        opacity=opacity,
        finish=finish,
    )
    return texture_path, uv_transform, rgba, uv_set, opacity, material


def _primitive_faces(glb, js: dict, buffers, world, primitive: dict,
                     material_index: int | None, image_paths, material):
    """Read one glTF triangle primitive into IngeTrazo world-space triangles."""
    if primitive.get("mode", 4) != 4:
        return None
    attributes = primitive.get("attributes", {})
    if "POSITION" not in attributes:
        return None

    positions = glb._accessor(js, buffers, attributes["POSITION"])[:, :3]
    if "indices" in primitive:
        indices = glb._accessor(js, buffers, primitive["indices"]).reshape(-1)
    else:
        indices = np.arange(len(positions), dtype=np.int64)
    if len(indices) % 3:
        raise ValueError("A glTF triangle primitive has an incomplete face.")
    indices = indices.astype(np.int64).reshape(-1, 3)
    if np.any(indices < 0) or np.any(indices >= len(positions)):
        raise ValueError("A glTF primitive contains an invalid vertex index.")

    zup = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    world_positions = positions @ world[:3, :3].T + world[:3, 3]
    triangles = (world_positions @ zup.T)[indices]

    texture_path, uv_transform, rgba, uv_set, opacity, _ = _material_for_gltf(
        glb, js, material_index, image_paths
    )
    uvs = None
    uv_key = f"TEXCOORD_{uv_set}"
    if texture_path and uv_key in attributes:
        uv = glb._accessor(js, buffers, attributes[uv_key])[:, :2]
        ox, oy, sx, sy = uv_transform
        uvs = (uv * (sx, sy) + (ox, oy))[indices]
    elif texture_path:
        texture_path = None

    face_attrs = []
    if texture_path and uvs is not None:
        uvw, valid = glb._triangle_uvws(
            triangles, uvs * (1.0, -1.0) + (0.0, 1.0)
        )
        for i in range(len(triangles)):
            attrs = {}
            if valid[i]:
                lengths = np.linalg.norm(uvw[i, :3]), np.linalg.norm(uvw[i, 4:7])
                attrs["texture"] = {
                    "path": texture_path,
                    "uvw": [float(value) for value in uvw[i]],
                    "sw": 1.0 / max(float(lengths[0]), 1e-9),
                    "sh": 1.0 / max(float(lengths[1]), 1e-9),
                }
                attrs["back"] = True
            if opacity is not None:
                attrs["opacity"] = opacity
            face_attrs.append(attrs or None)
    else:
        color = rgba[:3] if rgba else None
        attrs = {}
        if color is not None and color != [1.0, 1.0, 1.0]:
            attrs["color"] = color
            attrs["back"] = True
        if opacity is not None:
            attrs["opacity"] = opacity
        face_attrs = [dict(attrs) if attrs else None for _ in triangles]

    for index, attrs in enumerate(face_attrs):
        if material_index is not None:
            if attrs is None:
                attrs = {}
                face_attrs[index] = attrs
            attrs["mat"] = material.name
            attrs["_gltf_material_index"] = material_index
    return triangles, face_attrs


def _build_node_mesh(mesh: Mesh, primitives, glb, js, buffers, world,
                     image_paths, materials) -> int:
    """Build one node's mesh, retaining per-face material and texture attrs."""
    flat = []
    attrs = []
    for primitive in primitives:
        material_index = primitive.get("material")
        material = None
        if material_index is not None:
            if material_index < 0 or material_index >= len(js.get("materials", [])):
                raise ValueError("A glTF primitive references a missing material.")
            if material_index not in materials:
                materials[material_index] = _material_for_gltf(
                    glb, js, material_index, image_paths
                )[-1]
            material = materials[material_index]
        triangles = _primitive_faces(
            glb, js, buffers, world, primitive, material_index,
            image_paths, material,
        )
        if triangles is None:
            continue
        face_positions, face_attrs = triangles
        flat.append(face_positions.reshape(-1, 3))
        attrs.extend(face_attrs)

    if not flat:
        return 0
    positions = np.concatenate(flat, axis=0)
    vertices, inverse = mesh.bulk_weld(positions)
    mesh.add_faces_welded(
        vertices,
        inverse,
        np.full(len(attrs), 3, dtype=np.int64),
        np.ones(len(attrs), dtype=np.int64),
        attrs,
    )
    from formats.fuse import soften_smooth_edges

    if len(mesh.edges) <= 500_000:
        soften_smooth_edges(mesh)
    return len(attrs)


def load_gltf_groups(file_path: str | Path) -> tuple[list[Group], dict]:
    """Load a glTF/GLB scene as nested IngeTrazo groups with baked Z-up coordinates."""
    from formats import glb

    path = Path(file_path)
    if path.suffix.lower() not in {".gltf", ".glb"}:
        raise ValueError("Select a file with a .gltf or .glb extension.")
    js, buffers = glb.parse_container(path)
    for extension in js.get("extensionsRequired", []):
        if "draco" in extension.lower():
            raise ValueError("GLB files with Draco compression are not supported.")
    image_paths = glb._extract_images(js, buffers, path)
    placements = glb._node_transforms(js)
    if not placements:
        raise ValueError("The glTF file has no scene nodes to import.")

    node_data = js.get("nodes", [])
    groups = {}
    node_order = []
    face_count = 0
    materials = {}
    for node_index, world in placements:
        if node_index < 0 or node_index >= len(node_data):
            raise ValueError("A glTF scene references an invalid node.")
        node = node_data[node_index]
        mesh_index = node.get("mesh")
        if mesh_index is not None and (
            mesh_index < 0 or mesh_index >= len(js.get("meshes", []))
        ):
            raise ValueError(f"Node {node_index} references a missing mesh.")
        mesh_name = ""
        if mesh_index is not None:
            mesh_name = str(js["meshes"][mesh_index].get("name") or "")
        name = str(node.get("name") or mesh_name or f"Node {node_index + 1}")
        groups[node_index] = Group(Mesh(), name=name)
        node_order.append(node_index)
        if mesh_index is not None:
            face_count += _build_node_mesh(
                groups[node_index].mesh,
                js["meshes"][mesh_index].get("primitives", []),
                glb,
                js,
                buffers,
                world,
                image_paths,
                materials,
            )

    parents = {}
    for parent_index, node in enumerate(node_data):
        for child_index in node.get("children", []):
            if child_index in groups:
                parents.setdefault(child_index, parent_index)
    for parent_index, node in enumerate(node_data):
        if parent_index in groups:
            children = [
                groups[index]
                for index in node.get("children", [])
                if index in groups
            ]
            if children:
                groups[parent_index].adopt(children)

    roots = [
        groups[node_index]
        for node_index in node_order
        if node_index not in parents
    ]
    if not roots:
        raise ValueError("The glTF scene has no root nodes to import.")
    if not any(group.mesh.faces or group.mesh.edges for group in groups.values()):
        raise ValueError("The glTF scene contains no supported face or line geometry.")

    return roots, {
        "groups": len(groups),
        "faces": face_count,
        "materials": materials,
    }


def _visit_faces(groups):
    """Yield every face in imported groups, including nested placements."""
    for group in groups:
        yield from group.mesh.faces
        yield from _visit_faces(group.children or ())


def _register_imported_material(registry, material: Material) -> str:
    """Keep distinct glTF finishes even when names and paint are otherwise equal."""
    base = material.name or "Material"
    name = base
    suffix = 2
    while name in registry:
        existing = registry[name]
        if (
            existing.color == material.color
            and existing.texture == material.texture
            and existing.opacity == material.opacity
            and existing.finish == material.finish
        ):
            return name
        name = f"{base} ({suffix})"
        suffix += 1
    material.name = name
    registry[name] = material
    return name


class GltfImportCommand(Command):
    """Undoable group import that also registers its named material recipes."""

    def __init__(self, roots, materials) -> None:
        self._roots = list(roots)
        self._materials = dict(materials)
        self._import = SnapshotImport(self._add_groups)
        self._before_materials = None
        self._after_materials = None

    def _add_groups(self, scene) -> None:
        for group in self._roots:
            group.layer = getattr(scene, "active_layer", None)
            scene.groups.append(group)
        scene.selection.clear()
        for group in self._roots:
            scene.selection.add(group)
        scene.version = getattr(scene, "version", 0) + 1

    @staticmethod
    def _restore_registry(scene, registry) -> None:
        scene.materials.clear()
        scene.materials.update(registry)

    def do(self, scene) -> None:
        if self._after_materials is not None:
            self._import.do(scene)
            self._restore_registry(scene, self._after_materials)
            return

        self._before_materials = dict(scene.materials)
        self._import.do(scene)
        try:
            material_names = {}
            for index, material in self._materials.items():
                material_names[index] = _register_imported_material(
                    scene.materials,
                    Material(
                        name=material.name,
                        color=material.color,
                        texture=dict(material.texture) if material.texture else None,
                        opacity=material.opacity,
                        finish=material.finish,
                    ),
                )
            for face in _visit_faces(self._roots):
                index = face.attrs.pop("_gltf_material_index", None)
                if index is not None:
                    face.attrs["mat"] = material_names[index]
            self._after_materials = dict(scene.materials)
        except Exception:
            self._import.undo(scene)
            self._restore_registry(scene, self._before_materials)
            raise

    def undo(self, scene) -> None:
        self._import.undo(scene)
        self._restore_registry(scene, self._before_materials)


def import_gltf_with_groups(app) -> None:
    """Choose and import a glTF/GLB file as groups in the active document."""
    file_name, _selected_filter = QFileDialog.getOpenFileName(
        None,
        "Import glTF/GLB with Groups",
        "",
        "glTF Files (*.gltf *.glb)",
    )
    if not file_name:
        return

    try:
        roots, stats = load_gltf_groups(file_name)
    except Exception as exc:
        log.exception("Could not import glTF/GLB file %s", file_name)
        QMessageBox.critical(
            None,
            "Import Failed",
            f"Could not import the file:\n{exc}",
        )
        return

    viewport = getattr(app, "viewport", None)
    scene = getattr(viewport, "scene", None)
    history = getattr(viewport, "history", None)
    if scene is None or history is None:
        QMessageBox.critical(
            None,
            "Import Failed",
            "There is no active document ready to receive the model.",
        )
        return

    try:
        history.execute(GltfImportCommand(roots, stats["materials"]))
    except Exception as exc:
        log.exception("Could not add imported glTF/GLB groups to the document")
        QMessageBox.critical(
            None,
            "Import Failed",
            f"Could not add the model to the document:\n{exc}",
        )
        return

    refresh_warning = ""
    try:
        notify = getattr(viewport, "notify_scene_changed", None)
        if callable(notify):
            notify()
        viewport.update()
    except Exception as exc:
        log.exception("Model imported, but the viewport could not be refreshed")
        refresh_warning = f"\n\nWarning: the viewport could not be refreshed ({exc})."

    QMessageBox.information(
        None,
        "Import Complete",
        f"Groups: {stats['groups']}\nFaces: {stats['faces']}\n"
        f"Materials: {len(stats['materials'])}\n"
        "The file hierarchy has been preserved. Press Ctrl+Z to undo."
        f"{refresh_warning}",
    )


def setup(app) -> None:
    app.add_menu_action(
        "Import glTF/GLB with Groups…",
        lambda: import_gltf_with_groups(app),
        tip="Imports glTF and GLB files while preserving groups and node hierarchy.",
    )
    log.info("glTF Group Importer loaded (key=%s)", getattr(app, "key", "?"))
