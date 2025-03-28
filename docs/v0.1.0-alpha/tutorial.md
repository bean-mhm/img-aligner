This a step-by-step tutorial on how to get started with
[img-aligner](https://github.com/bean-mhm/img-aligner) v0.1.0-alpha. Below are
simple steps to align a few example images that come with img-aligner.

> [!IMPORTANT]
> This tutorial assumes you have fully read the [README](../../README.md) and
> have a basic understanding of the underlying algorithm. If you haven't read
> the README, expect to be confused. If the parameters don't make sense, you
> probaby haven't read the README patiently.

# Base & Target

The first step is to load a base image and a target image. The goal is to warp
and distort the base image to make it look similar to the target image.
img-aligner comes with a few example images in the `demo` directory which we'll
make use of.

1. In the _Controls_ tab, use the _Load Base Image_ button in the _IMAGES_
section to open a file dialog and choose `demo/images/1-house-base.exr`. Next,
hit _Load Target Image_ and choose `demo/images/1-house-target.exr`.

You'll notice two controls in the _IMAGES_ section named _Base Image Multiplier_
and _Target Image Multiplier_. We'll address these in later examples.

# Image Viewer

After loading the images, the _Image Viewer_ tab will be displaying the image
that's currently selected. You can choose to display another image at the top.
You can also adjust the exposure and the zoom level, check _flim_ to apply the
[flim](https://github.com/bean-mhm/flim) color transform on the image when
displaying, or check _Preview Grid_ to see a preview of the grid used for
warping.

# Grid Warper

The grid warper is the underlying construct that warps the base image and
calculates the difference between the warped image and the target image. We
can adjust grid warping settings in the _GRID WARPER_ section. If you hold the
cursor over a control's name, a tooltip may show up explaining what it does. For
this tutorial, we'll leave these intact and use the default values.

2. Hit _Recreate Grid Warper_ to prepare resources for grid warping. This will
switch the current image in the _Image Viewer_ to the _Difference Image_.

> [!NOTE]
> Every time you change the grid warper settings, the grid warper gets
> destroyed along with its images (warped / difference / cost) so you need to
> click on _Recreate Grid Warper_ again.

# Transform

We can use the sliders in the _TRANSFORM_ section to apply a linear transform to
the grid vertices to potentially make the difference smaller (which would make
the difference image darker).

If you keep switching between the base image and the target image in the
_Image Viewer_, you'll notice the target image has been slightly translated to
the right. You can adjust the _Offset_ parameter to compensate for this.

3. Adjust the _Offset_ until the difference image becomes darker.

> [!TIP]
> Hold \[Alt] while adjusting a drag control to slow down or \[Shift] to speed
> up.

# Optimization

The _OPTIMIZATION_ section lets us adjust optimization parameters, which contain
tooltips as well and are fairly straightforward. You can specify stop conditions
in the _STOP IF_ section.

> [!TIP]
> Hold \[Ctrl] and click on a slider to type in a value.

Once again, we won't change these settings for this
example.

4. Hit _Start Alignin'_ to start minimizing the cost (as explained in the
[README](../../README.md)).

Once optimization starts, a new section named _STATS_ will show up which will
display optimization statistics and a plot of the cost. Also, the image viewer
will switch to the difference image which is updated in realtime along with
the grid preview.

# Another Example

The next example uses two images from an exposure bracket, as explained in
the [README](../../README.md) (read it!).

5. Load `demo/images/2-exposure-1-over-28.exr` as the base image and
`demo/images/2-exposure-1-over-10.exr` as the target image.

As can be seen in the file names, the base image is a linear image captured in
1/28 seconds, and the target image has a light exposure time of 1/10 seconds,
so the target image is 2.8 times brighter than the base image.

Switch between the base and target images in the _Image Viewer_ and notice the
sudden jump in brightness. This is bad for optimization, so let's fix it.

6. In the _IMAGES_ section, set _Base Image Multiplier_ to 2.8.

> [!TIP]
> Check _flim_ in the _Image Viewer_ to apply a filmic color transform when
> displaying images.

If you switch between the images _now_, you'll notice that the brightness
doesn't change anymore, which is what we want.

7. Hit _Recreate Grid Warper_.

> [!IMPORTANT]
> When you load a new set of images, the grid transform settings are
> __kept the same__. If this is not what you want, you should reset the
> transform.

8. Hit _Reset_ in the _TRANSFORM_ section.

We don't need to change the transform for this set of images, but we'll modify
the optimization settings.

9. Play with _Warp Strength_ and related parameters in the _OPTIMIZATION_
section and observe how the warp strength plot changes.

10. Set _Warp Strength_ to 0.0002 and adjust the _Warp Strength Decay Rate_ to
about 0.001 to make the warp strength go down as the number of iterations
increases.

11. Set _Min Warp Strength_ to 0.0001 to prevent the warp strength from getting
too tiny.

12. Hit _Start Alignin'_ and observe mathematical beauty.

# Exporting

The _EXPORT IMAGES_ and _EXPORT METADATA_ sections let you export the warped
image, the difference image, or metadata to files.

# Misc

Apart from a UI scale control, the _Misc_ tab simply shows the app name and
version and contains a link to the GitHub page.

# Getting Better Results

img-aligner is designed to work with images that are already extremely similar,
i.e. the initial difference should be low. If the two images are so different
that most features don't overlap, img-aligner will do a pretty bad job. Below
are some methods to improve the results.

## Grid Transformation

For a better initial difference, you can adjust sliders in the _TRANSFORM_
section to apply a scale, rotation, and offset to the grid vertices to
potentially make the difference image darker. This might not work on all images.

## Images with Different Brightnesses

If one of the base or target images is brighter or darker than the other by a
linear factor (as is the case for exposure-bracketed images), adjust the
_Base / Target Image Multiplier_ in the _IMAGES_ section to compensate for that.

## Warp Strength Decay

By default, the warp strength is constant and has a tiny value. This works well
for images with tiny differences, but for larger differences, you can increase
both the _Warp Strength_ and the _Warp Strength Decay Rate_ which controls the
rate of decay of the warp strength. More precisely, the warp strength will be
scaled by `e^(-di)` where `d` is the decay rate and `i` is the number of
iterations.

A typical value for the decay rate could be 0.001. To avoid getting near zero
after decaying for too long, you can set the _Min Warp Strength_ value to clamp
the warp strength at a lower limit.

This decay lets us start with a higher warp strength to cancel out larger
differences between the images and slowly turn down the warp strength to work on
smaller differences.
