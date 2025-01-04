# What

This program allows you to warp an image to look like another, similar image.

# Why

The original goal was to perspective-align two or more photos with different
exposure times taken from the same scene with slighly different camera angles.
This can be used in __HDR fusion__ where we mix photos from the same scene with the
same camera angle but with different exposure levels by using brighter ones to
get details in the highlights (sky, headlights, etc.) and the darker photos to
get clean and noise-free shadows.

Remember that bright photos are typically
overexposed / clipped in the highlights but have clean shadows, and darker
ones maintain details in the highlights but are noisy in dark areas like shadows.
By combining them, we can get the best of both worlds.

This program does __not__ perform HDR fusion, it just helps you align your photos to avoid artifacts later.

# How

The algorithm uses grid warping to distort the starting image to roughly match
the target image. An evolution algorithm is used for optimization to minimize
the difference from the target image. Here's what the algorithm does in pseudocode:

```
in every iteration:
    randomly choose a grid point
    offset it by a gaussian distribution
    recalculate error from target image
    if new_error > old_error:
        undo the offset
```

For increased performance, grid warping and error calculation are performed on
the graphics processing unit (GPU) using the Vulkan API.

# Usage

1. Use the _Load Starting Image_ button in the _IMAGES_ section to open a file dialog and choose the image
you want to distort. Next, hit _Load Target Image_ to load the target image.

2. Adjust settings in the _SETTINGS_ section. This is optional.

3. Hit _Optimize_ to start the optimization process.

# Command Line

If you want to batch-process several images, you can call this program from a
command line or another program using the `--cli` argument to enable the command line interface.
If no other arguments are provided, a help text will be printed.

# How It's Built

This program is written in C++20 and uses the following libraries.

| Library | Used for |
|--|--|
| [GLFW](https://www.glfw.org/) | Window management |
| [beva](https://github.com/bean-mhm/beva) | Vulkan wrapper |
| [Dear ImGui](https://github.com/ocornut/imgui) | Graphical user interface |
| [NFD Extended](https://github.com/btzy/nativefiledialog-extended) | Native file dialogs |
| [OpenColorIO](https://opencolorio.org/) | Color management |
| [OpenImageIO](https://github.com/OpenImageIO/oiio) | Reading and writing images |

# Color Management

The program uses the [OpenColorIO](https://opencolorio.org/) library for color
management and [OpenImageIO](https://github.com/OpenImageIO/oiio) for loading
and saving images. It performs calculations in a linear color space using 32-bit
floating point values.

## OCIO Configs

An [OpenColorIO](https://opencolorio.org/) config typically contains an organized list of transforms and settings for color management. The following is what you typically find in an OCIO config (not in order), along with short descriptions. Note that this is my own understanding as a learner, so take it with a grain of salt.

| Element | Description |
|--|--|
| Reference Color Space | Not to be confused with the working color space, the reference color space is used to convert from any color space to any other color space. If we know how to go from color space *A* to the reference and vice versa, and we know how to go from color space *B* to the reference and vice versa, then we can easily go from *A* to *B* by first converting to the reference space, then to the target color space, *B*. |
| Color Space | A color space usually tells OCIO how to convert from and to the reference space by describing what transforms to use. It may also contain additional information on how data must be stored in said color space, a description, a list of aliases, and so on. |
| View | A view simply references a color space to be used. It may have a different name than the color space it references. |
| Display | A display contains a list of views that can be used with said display format. |
| Role | Roles are simply aliases for color spaces. As an example, the line `reference: Linear CIE-XYZ I-E` defines the reference color space. The `scene_linear` role generally defines the color space in which rendering and processing should be done. The `data` role defines the color space used for raw or generic data, A.K.A non-color data. The `color_picking` role defines what color space should be used in color pickers, and so on. Note that a program does not have to use these roles. For example, a program might always use `sRGB` in its color pickers regardless of the `color_picking` role. The `reference` role is an exception of this, as it is always used to convert between color spaces. |
| Look | A look is an optional transform that may be applied to an image before displaying it. |

The OCIO library lets us convert between color spaces, apply display/view transforms and looks, and so on.

Let's go through the different sections in the _Color Management_ panel.

## VIEW

Here we can alter how we *view* the image contained in the current slot by adjusting the exposure, the *Display* type, and the *View* transform, and optionally choosing an artistic *Look*. This does __not__ affect the pixel values of the image in any way, it just defines how the image is displayed.

## IMAGE IO

These settings define how images are imported and exported.

| Parameter | Description |
|--|--|
| Input | The interpreted color space when importing images in linear formats such as OpenEXR. The imported image will be converted from this color space to the working space. |
| Output | The output color space when exporting images in linear formats without a display/view transform. |
| Non-Linear | The interpreted color space when importing images in non-linear formats like PNG and JPEG. |
| Auto-Detect | If enabled, RealBloom will try to detect the color space when loading linear images, and discard the *Input* setting if a color space was detected. |
| Apply View Transform | If enabled, the current display/view transform will be applied when exporting linear images. The display/view transform is always applied to non-linear images. |
