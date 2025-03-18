# img-aligner

This program allows you to warp an image (the __base image__) to look
like another, already similar image (the __target image__).

# Why

My personal motivation for this project was to perspective-align two or more
photos with different exposure durations taken from the same scene and almost
the same camera angle. This can be used in __HDR fusion__ where we mix photos
from the same scene with different exposure levels to get a clean, noiseless
linear image that we can further work on (e.g. by applying a view transform,
commonly called a tone mapper). We can do this by using the darker images to
capture details in the highlights (sky, headlights, etc.) since the image is
dark enough to prevent overexposure and clipping, and we can use the brighter
photos to get clean and noise-free shadows. Again, darker images prevent
overexposure or clipping in bright areas, and brighter images prevent noise in
dark areas. By combining them, we can get the best of both worlds. I usually do
this with a custom shader node setup I've made in
[Blender](https://www.blender.org/).

Now, HDR fusion requires both the scene and the camera to be fully stationary
because there's a delay between each image we capture (not to mention motion
blur in images with higher exposure times). Using a tripod can help, but it's
not always an option, and sometimes I just need to pull out my phone and, using
[Open Camera](https://opencamera.org.uk/), press capture in the
"Exposure Bracket" mode which will automatically take 5 RAW (DNG) images with
varying exposure levels. Obviously, this introduces tiny movements in the camera
(the phone) simply from my hand movements, even if I hold my breath.

Despite being tiny, these movements are still large enough to ruin the
images for HDR fusion by introducing artifacts, so what I needed was a program
that aligns these "Exposure Bracket" images to fix my hand movements. Adobe
Photoshop has a similar feature, but it doesn't seem to work for images in
linear color spaces with 32-bit floating-point data, and I don't wanna rely on
paid Adobe software, or any Adobe software for that matter.

# How

The algorithm uses grid warping to distort the __base image__ to match the
__target image__. An evolution algorithm is used for optimization to minimize
the "cost". To calculate the cost, we render the per-pixel logarithmic
difference (between the warped base image and the target image) into what we call
the __difference image__ which we then downscale to a really tiny resolution
like 16x9 (called the __cost resolution__) and store it in the __cost image__.
The downscaling is done in a smooth way and samples all necessary pixels to
avoid aliasing.
Finally, we find the __maximum and average__ value in the pixels of the cost image.
The average (called the __average difference__) becomes our cost value, which is what we're trying to minimize.

The maximum (called the __maximum local difference__), however, is just there to make sure we don't introduce local
differences while decreasing the average difference. If the cost
resolution is 1x1, this will have no effect, but if the cost resolution is too high, it can slow down the
optimization. Note that, in each iteration, we compare the cost after
warping to the cost before warping (stored in the previous iteration). However,
for the maximum value, we only compare it to the initial maximum value that we got
in the beginning, and not the one from the previous iteration.

Here's what the algorithm looks like in every iteration:
1. Warp the grid vertices using a gaussian distribution with a random center,
  direction, and strength.  The ranges of the random values are calculated based
  on parameters.
2. Recalculate the cost (average difference) and the maximum value in the cost image.
3. If (cost > previous iteration's cost) or (maximum > initial maximum) the undo the warping.
4. Break the loop if stop conditions are met.

For increased performance, grid warping and cost calculation are performed at
a lower resolution (called the __intermediate resolution__) on the graphics
processing unit (GPU) using the Vulkan API.

# Color Spaces

Unlike typical images we might see on the internet which can only store RGB
(red, green, blue) values in the [0, 1] range, linear images allow any real
number (even negative) for the RGB values in their pixels.

img-aligner only supports OpenEXR images. It performs calculations in a linear
color space using 32-bit floating point values. All images are assumed to be
in Linear BT.709 (AKA Linear Rec. 709) or something similar, like
Linear BT.2020.

img-aligner always assumes your display device uses the sRGB standard. If you're
using a P3 or BT.2020 device, linear images that were originally intended to
work in BT.709 might look strongly vibrant on your display. This only affects
how you view images, not how they're processed or warped.

If you're curious, I really tried adding the
[OpenColorIO](https://opencolorio.org/) and
[OpenImageIO](https://github.com/OpenImageIO/oiio)
libraries for proper color management and image IO (like in
[RealBloom](https://github.com/bean-mhm/realbloom)), but they were painfully
hard to include and build with CMake, and I got errors after errors, so I gave
up.

# Usage

1. In the _Controls_ tab, use the _Load Base Image_ button in the _IMAGES_
section to open a file dialog and choose the image you want to distort. Next,
hit _Load Target Image_ to load the target image.

> If one of the images is darker or brighter than the other by a linear factor (as is the case for exposure bracketed images), you can adjust the _Base / Target Image Multiplier_ to compensate for that.

2. Adjust grid warping settings in the _GRID WARPER_ section and hit
_Recreate Grid Warper_.

3. Adjust optimizations settings in the _OPTIMIZATION_ section and hit
_Start Alignin'_ to start minimizing the cost.

4. Observe optimization statistics and a plot of the cost over time in the
_STATS_ section.

You can see the currently selected image in the _Image Viewer_ tab, and you can
choose to display another image. You can also adjust the exposure and the zoom
level, tick the _flim_ checkbox to apply the
[flim](https://github.com/bean-mhm/flim) transform on the image when displaying,
or tick _Preview Grid_ to see a preview of the grid used for warping.

# Command Line

To batch-process multiple images, you can call img-aligner from a terminal
or another program using the `--cli` argument to enable the command line
interface (CLI). If no other arguments are provided, a help text will be
printed.

# How It's Made

This program is written in C++20 and uses the following libraries.

| Library | Used for |
|--|--|
| [GLFW](https://www.glfw.org/) | Window management |
| [beva](https://github.com/bean-mhm/beva) | Vulkan wrapper |
| [Dear ImGui](https://github.com/ocornut/imgui) | Graphical user interface |
| [NFD Extended](https://github.com/btzy/nativefiledialog-extended) | Native file dialogs |
| [OpenEXR](https://openexr.com) | Reading and writing OpenEXR images |

# How to Build

This project uses CMake as its build system (they don't like it, but if it
works it works).

1. Make sure you've installed [CMake](https://cmake.org/) and proper C++ compilers. On Windows, you
can use [MSYS2](https://www.msys2.org/) which comes with CMake, GCC, mingw-w64,
and other useful tools and libraries.

2. Make sure the [Ninja](https://ninja-build.org/) build system is installed.

3. Create a `build` directory in the root directory of the repository and change
the working directory to it.
```bash
# delete if it already exists
rm -rf ./build

mkdir build
cd build
```

4. Generate CMake configuration with Ninja.
```bash
cmake -G "Ninja" ..
```
Make sure you have a stable internet connection so that unavailable packages
can be fetched online. You only need to regenerate this in certain cases, like
when you add or remove source files or modify `CMakeLists.txt`.

5. Build & Run.
```bash
# build in debug mode
cmake --build . --config Debug

# or release mode
cmake --build . --config Release

# run
./bin/img-aligner
```
