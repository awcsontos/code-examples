#!/usr/bin/env python
# coding: utf-8

# # P2 Feature Detection, Feature Matching and Image Warping
# 
# [The instructions are the same as last time.]

# ## P2.1 Harris Corners
# 
# Now that you've learned about Harris Corners, you're going to implement this procedure. For this question, **you should use the `light_cubes_base.png` and another image of your choosing**. The procedure for the Harris corner detection: 
# 
# 1. Compute the gradient at each point in the image
# 2. Compute the H matrix (or at least the elements of it) from the elements of the gradient (and a "smoothing" operation applied via a convolution with the weight matrix W)
# 3. Compute the scoring function (perhaps by computing the eigenvalues)
# 4. Find points with a large response (threshold f)
# 5. Find local maxima of f after thresholding (you will need to pick what you think is a "reasonable" threshold)
# 
# Where the Harris matrix $H$ is defined by
# $$ H = \begin{bmatrix} A & B \\ B & C \end{bmatrix} = \sum_{(x, y) \in W} w_{x, y} \begin{bmatrix}
# I_x^2 & I_x I_y \\ I_x I_y & I_y^2
# \end{bmatrix} $$
# and $I_x$ is the $x$-derivative of the image $I$, $I_y$ is the $y$-derivative of the image $I$ and $w$ is a weight matrix. Use the Sobel filters to compute the image derivatives. Use a convolution with the specified "weight matrix" W to compute the elements of the H matrix from the elements of the gradient multiplied together. You can use either definition of $f$ we introduced in class:
# 
# $$ f = min(\lambda_1, \lambda_2) $$
# $$ f = \frac{\text{det}(H)}{\text{tr}(H)} $$
# 
# I have provided you with a local-maxima-finding function for you to use.
# 
# **Note that you should be completing this solution without using a Harris Corner detector package: you should be computing the image derivatives and the Harris corners themselves yourself using filters you implement, applied using the `scipy` convolution functions.** Using external packages to check your solutions is acceptable, but your answer will no doubt be slightly different than those packages.
# 
# ### P2.1.1 Computing Harris Corners
# 
# **PLOTS: Generate a figure (a 2x2 image grid) that shows the following intermediate steps** during the computation of Harris Corners on the provided:
# 1. The x-derivative of the image, computed via the Sobel filter.
# 2. The "A" element of the H matrix for the whole image. To compute this, use the Sobel filter to compute the x-derivative and use a 5x5 mean filter to apply the effect of the window averaging (after squaring the x-derivative).
# 3. The scoring function $f$
# 4. The original image with dots showing the location of the detected corners. (Be sure to choose a *reasonable* threshold; you should expect to end up with on the order of a few dozen features. If you have thousands of features, your threshold is too low. If you have only one or two features, your threshold is too high.
# 
# For this first part, you should use a *uniform weight matrix of size 5x5* (`weights = np.ones((5, 5))/25`) when computing the scoring function. We will change this in the next part of the question. 
# 
# **QUESTION A:** (3 sentences) Are there any features in the Light Cubes image that you are surprised are *not* present? Highlight or discuss in words one or two regions of one of your images where features were detected that you did not expect or one or two regions you thought features might exist but do not. 
# 
# > **Answer:** They generally should be, since we understand this corner function pretty well. Seeing corners where there are no *physical* corners (only corners in the image).
# 
# In class, we derived the Harris matrix and a special scoring function related to the eigenvalues of that matrix so that we could measure the *cornerness* of the image. 
# 
# **PLOTS:** Show what happens if you used a scoring function $f = \text{tr}(H) = A + C$ (the trace of the H matrix)? Plot this alternative scoring function for the `light_cubes_base` image and plot the detected features computed using it.
# 
# **QUESTION B:**
# (2-4 sentences) What does this modified scoring function $\text{tr}(H)$ detect?  If we want to detect corners, why might we not want to use this scoring function? 
# 
# > **Answer:** This function detects edges as much as it does corners, since it's just the sum of the two eigenvalues. It is therefore not very effective at detecting corners alone.

# ### P2.1.2 Varying the Weight Matrix
# 
# In this part, we will see what happens when we use different weight matrices.
# 
# **PLOTS** Using only the `light_cubes_base` image, plot the score function $f$ (one of the ones we discussed in class) and the detected corners for each of the following weight functions:
# 
# 1. A uniform weight matrix of size 5x5 `weights = np.ones((5, 5))/(5**2)` (same as in the previous question).
# 2. A Gaussian weight matrix with $\sigma = 5$ (and size 25x25).
# 3. A uniform weight matrix of size 49x49 `weights = np.ones((49, 49))/(49**2)`
# 
# **QUESTION A:**
# (3-5 sentences) Discuss the differences between these three weight functions. In particular, how do the features change for the third kernel, which is significantly 'wider' than the others?
# 
# > *Answer:* The third kernel "smooths" out some of the regions in the middle of the frontward cube, and so some of the smaller features are not detected. The first two kernels have a somewhat similar response in terms of detected corners, but looking at the scoring function for each, you can see why it might be advantageous to use the Gaussian.
# 
# **QUESTION B:**
# (2 sentences) What happens to the score function if we were to use a 1x1 weight matrix $w = [1]$? Does this make sense? (Hint: think back to how we have defined and measure "cornerness"? Does it make sense to determine cornerness with a 1x1 window?)
# 
# > *Answer:* the scoring function is not well defined and in fact returns zero for all the pixels in the image. This should follow from the derivation of the operator itself, in which we were looking at the change over a window of pixels: if the window is only a 1x1 square, it's hard to meaningfully look for corners.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage
import scipy.ndimage.filters as filters

def load_image(filepath):
    """Loads an image into a numpy array.
    Note: image will have 3 color channels [r, g, b]."""
    img = Image.open(filepath)
    return (np.asarray(img).astype(float)/255)[:, :, :3]

def get_gaussian_filter(n, sigma=1):
    assert(n % 2 == 1)
    h = (n + 1)//2
    d = np.arange(h)
    d = np.concatenate((d[::-1], d[1:]))
    d = d[:, np.newaxis]
    d_sq = d**2 + d.T ** 2
    # Take the gaussian
    g = np.exp(-d_sq/2/(sigma**2))
    # Normalize
    g = g/g.sum().sum()
    return g

def get_local_maxima(data, threshold, do_return_values=False):
    # See: https://stackoverflow.com/a/9113227/3672986
    neighborhood_size = 3

    data_region_max = scipy.ndimage.maximum_filter(data, neighborhood_size)
    maxima = (data == data_region_max)
    data_min = scipy.ndimage.minimum_filter(data, neighborhood_size)
    maxima[data < threshold] = 0

    labeled, num_objects = scipy.ndimage.label(maxima)
    slices = scipy.ndimage.find_objects(labeled)
    x, y, r = [], [], []
    for dy, dx in slices:
        x_center = int(round((dx.start + dx.stop - 1)/2))
        x.append(x_center)
        y_center = int(round((dy.start + dy.stop - 1)/2))   
        y.append(y_center)
        r.append(data[y_center, x_center])
        
    if do_return_values:
        return np.array(list(zip(x, y))), r
    else:
        return np.array(list(zip(x, y,)))

# Some example code showing that it works
signal = np.random.rand(50, 50)
fig = plt.figure(figsize=(8, 8), dpi=150)
plt.imshow(signal, cmap='gray')
points = get_local_maxima(signal, threshold=0.95)
plt.plot(points[:, 0], points[:, 1], 'ro')
plt.show()

image = load_image("light_cubes_base.png")[:, :, 0]
plt.figure(dpi=300)
plt.imshow(image)
plt.show()
None


# In[19]:


# SOLUTION

def get_gaussian_filter(n, sigma=1):
    assert(n % 2 == 1)
    h = (n + 1)//2
    d = np.arange(h)
    d = np.concatenate((d[::-1], d[1:]))
    d = d[:, np.newaxis]
    d_sq = d**2 + d.T ** 2
    # Take the gaussian
    g = np.exp(-d_sq/2/(sigma**2))
    # Normalize
    g = g/g.sum().sum()
    return g

def get_harris_components(image, weights):
    sobel_x = np.array([
      [1, 0, -1],
      [2, 0, -2],
      [1, 0, -1]
    ])

    sobel_y = np.array([
      [1, 2, 1],
      [0, 0, 0],
      [-1, -2, -1]
    ])

    Ix = scipy.signal.convolve(
        image, sobel_x, mode='same')
    Iy = scipy.signal.convolve(
        image, sobel_y, mode='same')

    # Compute the Harris operator matrix
    w = weights
    A = scipy.signal.convolve(
        Ix * Ix, w, mode='same')
    B = scipy.signal.convolve(
        Ix * Iy, w, mode='same')
    C = scipy.signal.convolve(
        Iy * Iy, w, mode='same')
    return (A, B, C)

def get_harris_score(image, weights):
    A, B, C = get_harris_components(image, weights)
    det = A*C - B*B
    tr = A + C
    f = det/tr
    return f


image = load_image("light_cubes_base.png")[:, :, 0]
sigma = 50
weights = get_gaussian_filter(25, sigma=5)
weights = np.ones((5, 5))/25
# weights = np.ones((49, 49))/(49 ** 2)

f = get_harris_score(image, weights)
corners = get_local_maxima(f, threshold=0.01)


plt.figure(dpi=300)
plt.imshow(f)

plt.figure(dpi=300)
plt.imshow(image)
plt.plot(corners[:, 0], corners[:, 1], 'r.')

None


# In[20]:


# SOLUTION: Answers to C, D, E
# Using the alternate "trace" definition of the scoring
# function does a good job at detecting *all* edges, which 
# makes it a poor corner detector. I have included code 
# that computes and plots this function below:

def get_harris_score_trace(image, weights):
    A, B, C = get_harris_components(image, weights)
    return A + C

weights = get_gaussian_filter(201, sigma=5)
f = get_harris_score_trace(image, weights)
corners = get_local_maxima(f, threshold=0.001)

plt.figure(dpi=300)
plt.imshow(f)
plt.figure(dpi=300)
plt.imshow(image)
plt.plot(corners[:, 0], corners[:, 1], 'r.')

None


# ## P2.2 Multi-scale Blob Detection
# 
# Here, we will be building on the in-class breakout session to experiment with using the (normalized) Laplacian of Gaussian (LoG) filter to detect "blobs" in an image and their scale.
# 
# ### P2.2.1 Scale-Normalized Filter Response
# 
# In this question, I have provided you with a simple "circle image", in which a filled circle is placed at the center of a square image. Here, you know that the circle is the feature you are trying to detect and that its location is at the center, so the feature does not need to be *located*. Instead, you are asked to **find the radius of the circle** (the "blob feature" of interest). I showed some very-related plots in class, so your results should look quite similar to those.
# 
# The steps are detailed below:
# 1. First, you will need to define the LoG filter function `get_LoG_filter(kernel_size, sigma)` using either the in-class notes or [this resource](https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm). 
# 2. **PLOTS (IMAGES REQUIRED)** Once you have computed the normalized LoG filter, apply it at multiple scales to the sample circle image I have provided you with and **plot a few of the filtered images**.  (Remember to make your kernel size big enough compared to your kernel's $\sigma$ so that it is not cut off at the edges.)
# 3. **PLOTS (GRAPH REQUIRED)** Plot "filter response" (the value of the image after the filter is applied) at the center of the circle versus $\sigma$. Confirm that the peak of the filter response at the center of the circle occurs at the $\sigma$ we expect. (Recall that the peak $\sigma$ value does not correspond to the radius of the circle).
# 
# **QUESTION A:** (1 sentence)
# What is the relationship between the peak $\sigma$ and the circle's radius? 
# 
# > *Answer:* As discussed in class $r = \sqrt{2}\sigma$
# 
# 
# **Note:** If you use the *unnormalized Laplace of Gaussian Filter*, the maximal feature response will not occur where you expect. Be sure to use the correct filter function.

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

def get_circ_image(image_size, radius):
    """Create an image of width `image_size` with a circle 
    of radius `radius` in its center."""
    assert(image_size % 2 == 1)
    h = (image_size + 1)//2
    d = np.arange(h)
    d = np.concatenate((d[::-1], d[1:]))
    d = d[:, np.newaxis]
    d_sq = d**2 + d.T ** 2
    # Threshold by squared radius
    d_sq = (d_sq <= radius**2).astype(float)
    return d_sq

def get_LoG_filter(kernel_size, sigma):
    raise NotImplementedError("Your task is to implement the LoG filter.")

def apply_filter(signal, filt):
    """Apply a filter to an image; wrapper around scipy."""
    return scipy.signal.convolve2d(signal, filt, mode='same')

im_half_size = 25
fig = plt.figure()
circ_img = get_circ_image(2 * im_half_size + 1, radius=10)
plt.imshow(circ_img)

None


# In[5]:


# SOLUTION

def get_LoG_filter(kernel_size, sigma):
    assert(kernel_size % 2 == 1)
    h = (kernel_size + 1)//2
    d = np.arange(h)
    d = np.concatenate((d[::-1], d[1:]))
    d = d[:, np.newaxis]
    d_sq = d**2 + d.T ** 2
    # Implement the filter
    log = (1 - d_sq/2/(sigma**2)) * np.exp(-d_sq/2/(sigma**2)) / (sigma**2)
    return log

# What happens when we sweep as a function of sigma?
im_half_size = 25
fig = plt.figure()
sigmas = np.arange(2, 20, 0.1)
circ_img = get_circ_image(2 * im_half_size + 1, radius=10)
plt.imshow(circ_img)
responses = []
for sigma in sigmas:
    pass # Implement this loop.
    filt = get_LoG_filter(kernel_size=31, sigma=sigma)
    processed = apply_filter(circ_img, filt)
    responses.append(processed[im_half_size, im_half_size])
    
fig = plt.figure()
plt.plot(sigmas, responses)
print(f"Maximal response occurs at sigma = {sigmas[np.argmax(responses)]}")
print(f"Maximal response occurs at r = {np.sqrt(2)*sigmas[np.argmax(responses)]}")


# ###  P2.2.2 Annotating an Image with Multi-Scale Detections
# 
# Now, let's assume that we have an image with multiple features and we don't know either where they are or what their "radius" or "scale" is. Your goal here is to simultaneously detect features and estimate their characteristic scale.
# 
# For testing purposes, I have provided you with a simple image with two circles in it. Your task is to automatically identify where these "blobs" are and what their radius is. By the time you're done, you should be able to automatically detect the feature locations *and* their scale, producing images like the following: 
# 
# <img src="auto_feature_detection_result.png" width="400">
# 
# Similar to the Harris Features from the last exercise, the features occur at extrema (both maxima and minima) in image-space. **Before we compute multi-scale features, pick 3 or 4 values of $\sigma$ and for each, plot the following:**
# 1. **PLOTS (GRAPH REQUIRED)** The filter response (applying the scaled LoG filter for a particular $\sigma$ to the image function) and;
# 2. **PLOTS (IMAGE REQUIRED)** The location of the extrema plotted on top of the original image (just like the Harris Corner exercise from the previous programming assignment). To compute the corners, you may use the `get_local_maxima_3D` function I have provided below (the same as from the last question, yet for 3D data to include sigma).
# 
# Now, we can put everything together. The multi-scale features we care about exist at extrema in both image space *and* in scale space. This will require computing the "blob" feature response in both image-space and in scale space (by iterating through different sigma values). Features will exist at extrema of $f$ in both image space and scale space. Your code for multi-scale blob detection will look something like the following:
# 
# ```python
# response = np.zeros(
#     [image.shape[0], image.shape[1], sigmas.size]
# )
# for ii, sigma in enumerate(sigmas):
#    filt = get_LoG_filter(kernel_size, sigma)
#    feature_response = apply_filter_to_image(image, filt)
#    # Store the absolute value (both large positive 
#    # and negative responses correspond to features).
#    response[:, :, ii] = np.abs(feature_response)
#    
# features = get_local_maxima_3D(response)
# ```
# 
# Once you have computed the features, plot them as circles of the appropriate radius on top of three images: (1) the two-circle "test" image I have provided, (2) the "sunflower_field.jpg" image I have provided in this folder, and (3) a third image of your choosing.
# 
# [This resource](https://www.delftstack.com/howto/matplotlib/how-to-plot-a-circle-in-matplotlib/#matplotlib-patches-circle-method-to-plot-a-circle-in-matplotlib) will help you in drawing circles in matplotlib, but you can complete the function I have provided below if you prefer.

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.ndimage.filters as filters


def plot_circ_features(image, features, ax):
    ax.imshow(image, cmap='gray')
    for m in features:
        # if m[2] == sigmas.max():
        #     continue
        radius = None
        if radius is None:
            raise NotImplementedError()
        cir = plt.Circle((m[0], m[1]), radius, color='r', fill=False)
        ax.add_artist(cir)
        
def get_local_maxima_3D(data, threshold, sigmas, neighborhood_size=5):
    # See: https://stackoverflow.com/a/9113227/3672986
    data_region_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_region_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    maxima[data < threshold] = 0

    labeled, num_objects = scipy.ndimage.label(maxima)
    slices = scipy.ndimage.find_objects(labeled)

    features = []
    x, y = [], []
    for dy, dx, dz in slices:
        x_center = int(round((dx.start + dx.stop - 1)/2))
        y_center = int(round((dy.start + dy.stop - 1)/2))
        z_center = int(round((dz.start + dz.stop - 1)/2))
        features.append((x_center, y_center, sigmas[z_center]))
        
    return features
    
def compute_multi_scale_features(image, sigmas, threshold, window_size=11):
    raise NotImplementedError()

fig = plt.figure()
sigmas = np.arange(2, 20, 0.1)
circ_img_a = get_circ_image(2 * im_half_size + 1, radius=12)
circ_img_b = get_circ_image(2 * im_half_size + 1, radius=8)
circ_img = np.concatenate([circ_img_a, circ_img_b], axis=1)
plt.imshow(circ_img)

None


# In[7]:


# SOLUTION

import matplotlib.pyplot as plt

def get_LoG_filter(kernel_size, sigma):
    assert(kernel_size % 2 == 1)
    h = (kernel_size + 1)//2
    d = np.arange(h)
    d = np.concatenate((d[::-1], d[1:]))
    d = d[:, np.newaxis]
    d_sq = d**2 + d.T ** 2
    # Implement the filter
    log = (1 - d_sq/2/(sigma**2)) * np.exp(-d_sq/2/(sigma**2)) / (sigma**2)
    return log

def plot_circ_features(image, features, ax):
    ax.imshow(image, cmap='gray')
    for m in features:
        # if m[2] == sigmas.max():
        #     continue
        cir = plt.Circle((m[0], m[1]), np.sqrt(2)*m[2], color='r', fill=False)
        ax.add_artist(cir)

def compute_multi_scale_features(image, sigmas, threshold, window_size=11):
    response = np.zeros(
        [image.shape[0], image.shape[1], sigmas.size]
    )
    for ii, sigma in enumerate(sigmas):
        w = get_LoG_filter(201, sigma=sigma)
        feature_response = scipy.signal.convolve(
            image, w, mode='same')
        response[:, :, ii] = np.abs(feature_response)
    
    return get_local_maxima_3D(response, threshold, sigmas=sigmas,
                             neighborhood_size=window_size)


# In[8]:


# Plotting


# Show for the circle image
image_base = circ_img
sigmas = np.arange(1.0, 10.0, 0.2)
features = compute_multi_scale_features(image_base, sigmas, 1.2)
fig = plt.figure(figsize=(8, 8), dpi=300)
plot_circ_features(image_base, features, plt.gca())

# Feature response as a function of sigma
# Sunflower Image from: https://local12.com/news/local/deerfield-township-sunflower-field-attracts-shutterbugs
image_base = load_image("sunflower_field.jpg")[:, :, 0]
sigmas = np.arange(4.0, 80.0, 1.0)
features = compute_multi_scale_features(image_base, sigmas, 0.8)
fig = plt.figure(figsize=(8, 8), dpi=300)
plot_circ_features(image_base, features, plt.gca())

# Feature response as a function of sigma 
# (you may replace with an image you chose)
image_base = load_image("light_cubes_base.png")[:, :, 0]
sigmas = np.arange(4.0, 80.0, 1.0)
features = compute_multi_scale_features(image_base, sigmas, 0.4)
fig = plt.figure(figsize=(8, 8), dpi=300)
plot_circ_features(image_base, features, plt.gca())


# ## P2.3 Image Warping
# 
# In this question, we will pick up where we left off from the in-class breakout session. You are tasked with writing a `transform_image` function that takes as input an `image` and `transformation_matrix`. The function takes in an image and transforms it according to the `transformation_matrix`.
# 
# Using the `upsample_image` function I have provided below, write a new function `transform_image` that applies a transformation matrix to an image. You should feel free to use `scipy` for interpolation, **but you may not use `scipy` for anything other than the convolution, peak detection, and interpolation**.
# 
# > Note: you will notice in the comments in the provided function that there is a convention difference you need to keep track of: the `image` is stored in (row, column) coordinates, which is different from the transformation matrix, typically stored in (x, y) coordinates.
# 
# **PLOTS AND DESCRIPTIONS** Your goal, once this function is written, is to **implement the following transformation kernels, apply them to an image of your choosing, plot the image, and describe any surprising behavior of the kernels**:
# 
# 1. The identity: $$\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ \end{bmatrix}$$
# 2. A rotation by 30 degrees
# 3. A rotation by 30 degrees and translated to the center (you don't have to compute the center exactly; just pick some parameters that translate the image so that it is roughly centered even after rotation)
# 4. Scale by a factor of 2 along the x-axis
# 5. This kernel (which you should describe): $$\begin{bmatrix} 1 & -1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \\ \end{bmatrix}$$
# 
# **WRITEUP** Be sure to include the transformation matrices in your writeup.
# 
# You can change the domain of the output image by experimenting with `xi` and `yi` (defined in the `transform_image` function below). Using the rotated image you created above, modify `xi` and `yi` so that the entire image is contained in the output. 
# 
# **QUESTION A:**
# How do the changes in these vectors correspond to changes in the warped image? 
# 
# Finally, experiment with homography transforms, by modifying the bottom row of the transformation matrix. Define two different homography transforms, write out their matrices and display the results. 
# 
# **QUESTION B:**
# How do the two "bottom row" parameters control how the image is warped?

# In[9]:


import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal
from PIL import Image

def load_image(filepath):
    img = Image.open(filepath)
    return (np.asarray(img).astype(float)/255)[:, :, :3]

def upsample_image(image, target_shape, interp_fn):
    """This function is from P1 and is here to give you an idea 
    of what your 'transform_image' should look like."""
    # Initialize the upsampled image
    image_up = np.zeros(target_shape)
    
    # Define the new coordinates (using the [y, x] convention
    # since image matrices are defined [row, column])
    new_xs = np.linspace(0, image.shape[1]-1, 
                         target_shape[1], endpoint=True)
    new_ys = np.linspace(0, image.shape[0]-1, 
                         target_shape[0], endpoint=True)
    
    # Loop through coordinates and set the image values
    for ix, new_x in np.ndenumerate(new_xs):
        for iy, new_y in np.ndenumerate(new_ys):
            image_up[iy, ix] = interp_fn(image, new_x, new_y)
            
    return image_up

def transform_image(image, transformation_matrix):
    # Notice that because matrices are stored "rows, columns",
    # we need to flip the "shape" coordinates so that the transformation
    # matrix does what we expect. The other convention is also acceptable,
    # as long as one is consistent. In this function, the transformation
    # matrix is assumed to be in [x, y, w] coordinates, even though the image
    # is stored in row, column (y, x) coordinates.
    sh = image.shape
    x = np.arange(image.shape[1]).astype(float)
    y = np.arange(image.shape[0]).astype(float)
    
    # For now, the dimensions of the output image will
    # remain unchanged. You could modify xi and yi to 
    # change the domain of the output image.
    xi = np.arange(image.shape[1]).astype(float)
    yi = np.arange(image.shape[0]).astype(float)

    # Perform the transformation
    image_fn = scipy.interpolate.interp2d(x, y, image, fill_value=0)
    # Note: you can use this 'image_fn' to perform interpolation instead
    # of your implementation: new_val = image_fn(new_x, new_y)
    transformed_image = np.zeros((len(yi), len(xi)))
    # TODO: Loop through all pixels in new 'transformed_image' and set their values
    raise NotImplementedError()
    return transformed_image


# In[10]:


# SOLUTION

## P2.3S Image Warping

# **NOTE**: There are a number of reasons 
# that it is difficult to determine the correct 
# direction of rotation. As such, I have given 
# full credit to both positive and negative rotation 
# for each response. In fact, since it was not 100% clear 
# when I was talking about the forward and the negative 
# transformation, I have given full credit to both 
# solutions for all transformations.
# 
# My solution for `transform_image` is as follows:

def transform_image(image, transformation_matrix):
    # Notice that because matrices are stored "rows, columns",
    # we need to flip the "shape" coordinates so that the transformation
    # matrix does what we expect. The other convention is also acceptable,
    # as long as one is consistent. In this function, the transformation
    # matrix is assumed to be in [x, y, w] coordinates, even though the image
    # is stored in row, column (y, x) coordinates.
    sh = image.shape
    x = np.arange(image.shape[1]).astype(float)
    y = np.arange(image.shape[0]).astype(float)
    
    # For now, the dimensions of the output image will
    # remain unchanged. Modify xi and yi to change the
    # domain of the output image.
    xi = np.arange(image.shape[1]).astype(float)
    yi = np.arange(image.shape[0]).astype(float)

    # Perform the transformation
    image_fn = scipy.interpolate.interp2d(x, y, image, fill_value=0)
    transformed_image = np.zeros((len(yi), len(xi)))
    for ii in range(len(xi)):
        for jj in range(len(yi)):
            ct = np.matmul(transformation_matrix, 
                           [[xi[ii]], [yi[jj]], [1]])
            new_x = ct[0]/ct[2]
            new_y = ct[1]/ct[2]
            transformed_image[jj, ii] = image_fn(new_x, new_y)
            
    return transformed_image


# In[11]:


## SOLUTION

image_base = load_image("light_cubes_sm.png")[::2, ::2, 0]
plt.figure()

# The Identity (not shown)
H = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

# Rotation by 30 degrees
plt.subplot(2, 2, 1)
th = np.pi * 30 / 180
H = [[ np.cos(th), np.sin(th), 0],
     [-np.sin(th), np.cos(th), 0],
     [0, 0, 1]]
plt.imshow(transform_image(image_base, H))

# Rotation and center
plt.subplot(2, 2, 2)
c_base = np.array([image_base.shape[1]/2, image_base.shape[0]/2,1])
c = np.array(H) @ c_base
H = [[ np.cos(th), np.sin(th), c_base[0]-c[0]],
     [-np.sin(th), np.cos(th), c_base[1]-c[1]],
     [0, 0, 1]]
plt.imshow(transform_image(image_base, H))

# Scale by 2 along the x-axis
plt.subplot(2, 2, 3)
H = [[0.5, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
plt.imshow(transform_image(image_base, H))

# Scale [shrink by sqrt(2)] + Rotation (-45 degrees)
plt.subplot(2, 2, 4)
H = [[1, -1, 0],
     [1, 1, 0],
     [0, 0, 1]]
plt.imshow(transform_image(image_base, H))

None


# > **ANSWER A**: Changing the values of `xi` and `yi` in the warping function changes the range of the output image. By increasing this range, you can fit the entire image into the output. The following modification would be sufficient:
# > ```python
# > xi = np.arange(-image.shape[1], 2*image.shape[1]).astype(float)
# > yi = np.arange(-image.shape[0], 2*image.shape[0]).astype(float)
# > ```
# > 
# > **ANSWER B**: I have included below two matrices that have non-zero "bottom row" elements that control perspective/non-affine transformations. Both create trapezoidal images out of the original square image. A positive value in the first coordinate stretches down the bottom-right corner. Similarly a positive value in the center coordinate, stretches the bottom right corner to the right. Both are a result of the normalization that occurs after multiplication, when we project back into homogeneous coordinates. Here is the code for the plotting:

# In[12]:


## SOLUTION

# H matrix a
plt.subplot(2, 1, 1)
H = [[ 1.5, 0, 0],
     [0, 1.5, 0],
     [0.001, 0, 1]]
plt.imshow(transform_image(image_base, H))

plt.subplot(2, 1, 2)
H = [[ 1.5, 0, 0],
     [0, 1.5, 0],
     [0, 0.001, 1]]
plt.imshow(transform_image(image_base, H))


# ## P2.4 Some Simple Feature Descriptors
# 
# In class you learned about some different strategies for describing features; now we're going to implement them and see how well they perform under different image transformations. Here are the relevant slides from class for the four different descriptors you will be implementing for this problem (exact vector comparison, an x-gradient binary descriptor, a color/brightness histogram, a spatial histogram):
# 
# <img src="descriptors_overview.png" width="600">
# 
# Compute some of the simple image descriptors we discussed in class  and see how effective they are at matching features across different image transformations. To locate the features, use your Harris Corner detector from question P2.1. I have provided code that computes the feature descriptors for small image patches surrounding each of your features using three of these strategies. I have also included code under `Descriptor Matching Plotting Code` that plots the matches for a set of test images.
# 
# **TASK** Implement the `get_corners` function (in the code block named `Computing and Matching Descriptors` using your Harris Corner detection solution from P2.1. If that function is implemented properly, you should be able to generate plots showing match quality.
# 
# **FIGURES** Once you have implemented `get_corners` the code below should generate plots for the three different descriptors. Include all three in your writeup.
# 
# **QUESTION A** Which feature descriptor performs poorly on the image `img_contrast`? Explain why this descriptor performs worse than the others.
# 
# **QUESTION B** Which feature descriptor performs best on the image `img_transpose`? Explain why this descriptor performs better than the others.

# In[13]:


## Helper code for plotting
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal
from PIL import Image

def load_image(filepath):
    img = Image.open(filepath)
    return (np.asarray(img).astype(float)/255)[:, :, :3]

def visualize_matches(img_a, img_b, matches, ax=None, title=None):
    """Visualize matches between two images. Matches is a list
    such that each element of the list is a 4-element tuple of
    the form [x1, y1, x2, y2]."""
    if ax is None:
        # Create a new axis if none is provided
        fig = plt.figure(dpi=300)
        ax = plt.gca()
    
    # Helper variables
    sa = img_a.shape
    sb = img_b.shape
    sp = 40
    
    # Merge the images and plot matches
    merged_imgs = np.zeros(
        (max(sa[0], sb[0]), sa[1]+sb[1]+sp),
        dtype=float)
    merged_imgs[0:sa[0], 0:sa[1]] = img_a
    merged_imgs[0:sb[0], sa[1]+sp:] = img_b
    ax.imshow(merged_imgs)
    
    for m in matches:
        ax.plot([m[0], m[2]+sa[1]+sp], 
                [m[1], m[3]],
                'r', alpha=0.7)
    
    if title is not None:
        ax.set_title(title)

# An example of match visualization
# Stored: [x1, y1, x2, y2]
img_base = load_image('tr_base.png')[:, :, 0]
matches = [
    [1, 3, 1, 3],
    [10, 13, 10, 13],
    [100, 20, 100, 20],
    [30, 100, 30, 100]
]
visualize_matches(img_base, img_base, matches)


# In[14]:


## Computing and Matching Descriptors

# Compute images for matching
img_base = load_image('light_cubes_sm.png')[:, :, 0]
img_contrast = img_base ** (0.4)
img_highres = load_image('light_cubes_base.png')[300:300+img_base.shape[0],
                                                 400:400+img_base.shape[1], 0]
img_transpose = img_base.T


# TASK: you need to implement Harris Corner Detection
# Set the default 'threshold' to your satisfaction.
def get_corners(image, threshold=0.01):
    """Note: this function assumed to return corners 
    as (x, y) coordinates *not* as (row, column) 
    coordinates."""
    raise NotImplementedError()

def get_features_with_descriptors(image,
                                  corners,
                                  compute_descriptor_fn, 
                                  patch_half_width=7):
    features = []
    for c in corners:
        patch = image[c[1]-patch_half_width:c[1]+patch_half_width+1,
                      c[0]-patch_half_width:c[0]+patch_half_width+1]
        
        # Remove patches too close to the edge
        if patch.size < (2*patch_half_width + 1) ** 2:
            continue
        features.append({
            'x': c[0],
            'y': c[1],
            'patch': patch,
            'descriptor': compute_descriptor_fn(patch),
        })
    
    return features


# Define the various descriptors

def compute_descriptor_match(patch):
    return (patch - np.mean(patch)) / np.std(patch)

def compute_descriptor_binary_x(patch):
    return 2 * ((patch[:-1] - patch[1:]) > 0).astype(float) - 1

def compute_descriptor_hist(patch):
    return np.sqrt(np.histogram(patch, bins=8, range=(0.0, 1.0))[0])


# Define the matching functions

def compare_descriptors(fa, fb):
    return np.sum(fa['descriptor'] * fb['descriptor'])

def compute_feature_matches(fsa, fsb):
    # First compute the strength of the feature response
    sims = np.zeros((len(fsa), len(fsb)), dtype=float)
    for ii, fa in enumerate(fsa):
        for jj, fb in enumerate(fsb):
            sims[ii, jj] = compare_descriptors(fa, fb)

    # Now compute the matches
    matches = []
    for ii in range(len(fsa)):
        mi = np.argmax(sims[ii])
        if not ii == np.argmax(sims[:, mi]):
            continue
        match_score = sims[ii, mi]
        matches.append([fsa[ii]['x'],
                        fsa[ii]['y'],
                        fsb[mi]['x'],
                        fsb[mi]['y']])

    return matches


# In[15]:


# SOLUTION: (relies on my functions from above) 
def get_corners(image, threshold=0.01):
    weights = get_gaussian_filter(201, sigma=5)
    f = get_harris_score(image, weights)
    return get_local_maxima(f, threshold=threshold)


# In[16]:


# Descriptor Matching Plotting Code
for (fn, title) in [(compute_descriptor_match, "Match"),
        (compute_descriptor_binary_x, "Binary (x)"),
        (compute_descriptor_hist, "Histogram")]:
    plt.figure(figsize=(12, 6))
    
    corners = get_corners(img_base)
    fsa = get_features_with_descriptors(img_base, corners, fn)
    
    for ind, image_comp in enumerate(
            [img_base, img_contrast, img_highres, img_transpose]):
    
        ax = plt.subplot(2, 2, ind+1)
        corners = get_corners(image_comp)
        fsb = get_features_with_descriptors(image_comp, corners, fn)
        matches = compute_feature_matches(fsa, fsb)
        visualize_matches(img_base, image_comp, matches, ax, title)


# ## [ANSWERS] P2.4 Data Generation Code
# 
# > (Plots and my implementation of `get_corners` shown above)
# > 
# > **Question A**: The histogram descriptor performs the most poorly on  `img_contrast`. This is expected behavior because changing the brightness and contrast of the image directly changes the histogram of the surrounding patch. By contrast, the other two descriptors either compensate for this shift (the `match` descriptor is scaled and normalized) or are unaffected by it (the sign of the derivative is also unchanged).
# > 
# > **Question B**: The *histogram* descriptor is the best performing on `img_transpose` because it is unaffected by reflections and rotations. The other two descriptors are both *very* impacted by the transformation and perform quite poorly.

# In[17]:


## SOLUTION: Data Generation
import pickle

img_base = load_image('light_cubes_sm.png')[:, :, 0]
img_contrast = img_base ** (0.4)
img_highres = load_image('light_cubes_base.png')[300:300+img_base.shape[0],
                                                 400:400+img_base.shape[1], 0]
img_transpose = img_base.T

def get_corners_top_n(image, threshold=0.01, N=8):
    weights = get_gaussian_filter(201, sigma=5)
    f = get_harris_score(image, weights)
    cs, rs = get_local_maxima(f, threshold, True)
    crs = list(zip(cs, rs))
    crs = sorted(crs, key=lambda x: x[1], reverse=True)
    return np.array(list(zip(*crs))[0][:N])

data = {
    'img_base': img_base,
    'corners_base': get_corners_top_n(img_base),
    'img_contrast': img_contrast,
    'corners_contrast': get_corners_top_n(img_contrast),
    'img_highres': img_highres,
    'corners_highres': get_corners_top_n(img_highres),
    'img_transpose': img_transpose,
    'corners_transpose': get_corners_top_n(img_transpose),
}

pickle.dump(data, open('breakout_descriptors_data.pickle', 'wb'))


# In[18]:


data = pickle.load(open('breakout_descriptors_data.pickle', 'rb'))

img_base = data['img_base']
corners_base= data['corners_base']
img_contrast= data['img_contrast']
corners_contrast = data['corners_contrast']
img_highres = data['img_highres']
corners_highres = data['corners_highres']
img_transpose = data['img_transpose']
corners_transpose = data['corners_transpose']

# Descriptor Matching Plotting Code
def plot_matches_for_descriptor(descriptor_fn):
    plt.figure(figsize=(12, 6))
    
    corners = corners_base
    fsa = get_features_with_descriptors(img_base, corners, descriptor_fn)
    
    for ind, (image_comp, corners) in enumerate(zip(
            [img_base, img_contrast, img_highres, img_transpose],
            [corners_base, corners_contrast, corners_highres, corners_transpose])):
        ax = plt.subplot(2, 2, ind+1)
        fsb = get_features_with_descriptors(image_comp, corners, fn)
        matches = compute_feature_matches(fsa, fsb)
        visualize_matches(img_base, image_comp, matches, ax, title)


# In[ ]:




