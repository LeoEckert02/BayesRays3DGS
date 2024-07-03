# Modified gsplat Library

## Overview

This gsplat code is based on gsplat library v1.1.0 (https://github.com/nerfstudio-project/gsplat).

## Modifications

1. Main change in `rasterize_to_pixels_fwd_kernel` function:
   - Added uncertainty calculations during rendering.
   - New input: `uncertainties` for each Gaussian.
   - New output: `render_uncertainties` for each pixel.

2. Wrapper functions updated:
   - Added `uncertainties` input parameter to relevant wrapper functions.

All other code remains unchanged from the original gsplat v1.1.0.

## Acknowledgements

Credit to the original gsplat creators and contributors.

## License

Subject to the original gsplat license.