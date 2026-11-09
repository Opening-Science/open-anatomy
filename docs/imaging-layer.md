# Imaging layer

**Status: prototype.** `scripts/imaging.py` builds it for one reference body.
Nothing in TwinLLM or OpenTwin loads it yet.

## What it is

The imaging layer shows a reference body's inside the way a scan does: flat
slices through the body, with the segmented structures tinted. Peter calls it
the "MRI layer". The source is CT, not MRI, because every open, commercially
usable scan with matching segmentations is CT (the TotalSegmentator dataset,
CC BY 4.0). The viewer must label it as CT.

Each reference body is built from one CT case: s0903 for the male and s0859
for the female. The imaging layer slices that same CT, so a slice through the
body cuts the same liver, kidneys and bones that the 3D layers show.

## What the prototype makes

`python scripts/imaging.py <body> [--window soft|bone]` writes, into
`data/imaging/<body>/`:

| File | Content |
| --- | --- |
| `axial.png`, `coronal.png`, `sagittal.png` | one slice each, windowed for soft tissue (W400/L40) or bone (W1800/L400); each segmented structure is tinted 38% towards its display colour (converted from linear to sRGB first, so tints match the meshes); air outside the body is transparent |
| `axial@2x.png`, `coronal@2x.png`, `sagittal@2x.png` | the same slices at twice the size, for display only: the colour is Lanczos-3 upsampled on premultiplied alpha and clamped to the range of the body voxels around each source voxel, so it has no dark fringe and no ringing; the edge is anti-aliased: the native mask is pulled in by one source voxel (its outer voxel is often air at the skin) and softened by a small Gaussian, and colour is taken from inside that, so the edge shows no staircase and no dark rim; the native PNGs stay the ones to measure on |
| `<body>.imaging.glb` | three double-sided, unlit quads (`KHR_materials_unlit`) carrying the 2x PNGs, with `alphaMode` MASK and `alphaCutoff` 0.5 (see below) |
| `imaging.json` | each slice's plane, voxel index, pixel size, SHA-256 (native and display), the structures it cuts (by UBERON or FMA ID) and its corners in metres |

The display PNGs are filtered in integer arithmetic, with the Lanczos weights
in fixed point (steps of 1/4096), and the edge's blur is a sum of shifted
copies in a fixed order, so their SHA-256 does not depend on the BLAS build or
the thread count. The filter weights themselves are computed in floating point
once; a machine whose `sin` or `exp` differs in the last bit could only change
a byte if a value sat within that bit of a rounding boundary.

**The display shows slightly less than the native slice.** Pulling the mask in
by one source voxel removes the dark rim of air at the skin, but it also
removes any part of the body two voxels wide or less and widens every gap by
two voxels. For the trunk that is only the skin's outer voxel; for thin
features (a finger of tissue, a narrow gap between arm and trunk) the display
PNG and the GLB show less than the native PNG, which keeps every voxel. Measure
on the native PNGs.

**Why MASK and not BLEND.** The three quads cut through each other. Viewers
sort BLEND materials per object, not per pixel, and three.js's GLTFLoader
turns depth writes off for them, so with BLEND one plane is drawn over another
wrongly wherever they cross. MASK is drawn with the depth buffer like an opaque
surface, so the crossings are right in every viewer. The soft edge is kept in
the texture's alpha: under bilinear filtering its 0.5 contour runs smoothly
(no voxel staircase), and a viewer that turns on alpha-to-coverage with MSAA
(three.js: `material.alphaToCoverage = true`) gets the anti-aliased edge too.

The GLB loads together with `<body>.reference.glb`: the quads sit where their
slices lie in the fitted body. They use the trunk fit from
`blender/assemble.py` but not the shoulder-girdle step, so near the shoulders
the bones in 3D can stand a few centimetres off their slice. A spot check on
the male body puts the slices' liver, left kidney and aorta within 0.4 mm of
the meshes (bounding-box centres).

### Where the slices are cut

- **Axial:** at the kidneys' mean height.
- **Coronal:** through the kidneys' mean depth, which shows both kidneys, the
  liver, spleen, spine and aorta in one plane.
- **Sagittal:** on the spine's midline.

### Orientation

The textures follow radiology's usual layout: the body's right is on the
image's left, superior is at the top, and in the axial slice anterior is at
the top. The quads' corners are computed from the CT's affine with the same
row and column order, so the texture lands the right way round in 3D whatever
the image layout. `scripts/imaging_check.py` checks this on a synthetic
phantom stored in a non-canonical voxel order: markers on the left, front and
top must land on the expected side of each image, and each marker's pixel must
map through the quad back to the marker's position.

### Crop and privacy

Every slice is cropped to the trunk outline (TotalSegmentator's
`body_trunc`), from 5 cm below the femurs' top (the head or the greater
trochanter, whichever is higher; a case without both femurs is refused) up.

A head guard (`face_free_top`) keeps faces out:

- `segment.py --set reference` also segments the skull and brain. They are
  used only by the guard, and a case without those masks is refused, so
  absence of a head is shown rather than assumed.
- If the skull or brain is in the scan, nothing above 2 cm below its lowest
  voxel reaches an image. A bowed head brings the jaw down level with C7, so
  the top of C7 alone is not a safe cut.
- If any of C1–C6 is in the scan, nothing above the top of C7 reaches an
  image. Without C7, the cut is 2 cm below the lowest of them.
- The slices stop at the lowest of these cuts. 2 cm is turned into slices
  from the scan's own slice spacing (14 at 1.5 mm, 25 at 0.8 mm, 7 at 3 mm),
  rounded up.
- A case whose head comes within 2 cm of the scan's bottom slice is refused:
  there would be no trunk below the cut to show.
- `scripts/imaging_check.py` tests these rules on synthetic masks in CI.

The male case s0903 shows why the skull rule matters. His head is bowed, and
TotalSegmentator finds jaw and skull voxels anteriorly at C7's height, so his
slices stop at axial index 379, about 2 cm below them. The female case s0859
has 64 skull voxels, a likely false positive in a scan that starts at T2, and
the guard cuts below them anyway. The source is an open, de-identified
research dataset; no scan of any identified person is used.

## Licence

The slices are derived from the TotalSegmentator dataset, CC BY 4.0, so when
they are released they go under CC BY 4.0 in `assets/cc-by/`, with the
dataset's credit line (see [NOTICE](../NOTICE)). Each GLB node already carries
`"licence": "CC-BY-4.0"`. For now they are built into the gitignored `data/`
folder like every other output.

## Next

1. **A slider, not three fixed slices.** Ship a downsampled volume (for
   example 3 mm, 8-bit, windowed, cropped to the trunk) as a texture atlas or
   KTX2 3D texture, so the viewer can move each plane through the body. At 3
   mm the male trunk is about 100 × 60 × 130 voxels, under 1 MB.
2. **Follow the girdle step,** so the upper slices line up with the fitted
   shoulders, or say plainly that slices show the scan as acquired.
3. **Labels on hover:** the viewer can map a pixel back to a structure through
   the label volume, using the same IDs as the 3D nodes.
4. **MRI-like contrast?** Restyling CT to look like MRI would invent contrast
   the data does not hold. The proposal is not to do that, and to keep "MRI
   layer" as the name of the view only.
