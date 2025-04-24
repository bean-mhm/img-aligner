This directory contains a number of exposure-bracketed images (starting with
`EXPR`), a Python script named `align-bracket.py` that uses img-aligner's CLI to
align the images, and a Blender file `exposure-fusion.blend` for combining the
aligned images into a single 32-bit linear image and minor post processing.

If you run the script and use the Blender file to exposure fuse (A.K.A HDR fuse)
the aligned images, you'll get something like `fused.png`.
Notice how parts of the image that have moved throughout the capture duration
(like the flags and the people) have artifacts around them. This is why it's so
important for both the camera and the subjects to stay still during an exposure
bracket.

> [!IMPORTANT]
> The Blender file was designed with
> [grace v1.1](https://github.com/bean-mhm/grace) as the
> [OpenColorIO](https://opencolorio.org/) config. You can switch to
> another OCIO config in Blender by renaming the directory
> ```.../Blender X.X/X.X/datafiles/colormanagement```
> to something like `colormanagement.backup` and making a new directory named
> `colormanagement`, then copying the contents of your new config (like grace)
> to that directory. You'll also need to close and reopen Blender.
