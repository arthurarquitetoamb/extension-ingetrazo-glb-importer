# glTF/GLB Group Importer for IngeTrazo

This plugin imports `.gltf` and `.glb` files as native IngeTrazo groups. It
preserves the file's node hierarchy, converts glTF's Y-up coordinates to
IngeTrazo's Z-up coordinates, and selects the imported groups for review.
Named materials are registered in the document with their colors, textures,
transparency, and finish when available. The import is a single undoable
history operation and can be reverted with `Ctrl+Z`.

<img width="615" height="263" alt="Groups" src="https://github.com/user-attachments/assets/022bc330-da4e-4c53-a13e-6d9b0179c1db" />


## Installation

Copy `gltf_group_importer.py` to the IngeTrazo user plugins folder:

```text
%APPDATA%\ingetrazo\plugins\
```

Restart IngeTrazo. The command will appear in the **Extensions** menu as
**Import glTF/GLB with Groups…**.

## Limitations

- Uses IngeTrazo's built-in glTF reader and requires no additional dependencies.
- Imports triangle primitives. Points, lines, and non-triangle drawing modes
  are not imported.
- Files requiring Draco compression are not supported. The importer preserves
  material names, colors, textures, and `BLEND` transparency; textured
  materials use the face UV mapping. PBR extensions not recognized by
  IngeTrazo are not reproduced.
- The glTF file's own unit system is used; the plugin does not convert units.
