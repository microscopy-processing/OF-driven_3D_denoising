import numpy as np
import scipy.ndimage
import cv2

def compute_gaussian_kernel(sigma=1):
  number_of_coeffs = 3
  number_of_zeros = 0
  while number_of_zeros < 2 :
    delta = np.zeros(number_of_coeffs)
    delta[delta.size//2] = 1
    coeffs = scipy.ndimage.gaussian_filter1d(delta, sigma=sigma)
    number_of_zeros = coeffs.size - np.count_nonzero(coeffs)
    number_of_coeffs += 1
  return coeffs[1:-1]

ofca_extension_mode = cv2.BORDER_REPLICATE

def warp_slice(reference, flow):
  height, width = flow.shape[:2]
  map_x = np.tile(np.arange(width), (height, 1))
  map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
  map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
  return cv2.remap(reference, map_xy, None, interpolation=cv2.INTER_LINEAR, borderMode=ofca_extension_mode)

def get_flow(reference, target, l, w):
  flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=None, pyr_scale=0.5, levels=l, winsize=w, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
  #flow = np.zeros((reference.shape[0], reference.shape[1], 2), dtype=np.float32)
  return flow

def filter_over_Z(tomogram, kernel, l, w):
  filtered_tomogram = np.zeros_like(tomogram).astype(np.float32)
  shape_of_tomogram = np.shape(tomogram)
  padded_tomogram = np.zeros(shape=(shape_of_tomogram[0] + kernel.size, shape_of_tomogram[1], shape_of_tomogram[2]))
  padded_tomogram[kernel.size//2:shape_of_tomogram[0] + kernel.size//2, :, :] = tomogram
  Z_dim = tomogram.shape[0]
  for z in range(Z_dim):
    tmp_slice = np.zeros_like(tomogram[z]).astype(np.float32)
    for i in range(kernel.size):
      if i != kernel.size//2:
        flow = get_flow(padded_tomogram[z + i], tomogram[z], l, w)
        #flow = get_flow(padded_tomogram[z + i - kernel.size//2], padded_tomogram[z - kernel.size//2], l, w)
        #flow = get_flow(padded_tomogram[z - kernel.size//2], padded_tomogram[z + i - kernel.size//2], l, w)
        #OF_compensated_slice = warp_slice(padded_tomogram[z + i - kernel.size//2], flow)
        OF_compensated_slice = warp_slice(padded_tomogram[z + i], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
      else:
        # No OF is needed for this slice
        #tmp_slice += tomogram[z - kernel.size//2, :, :] * kernel[kernel.size // 2]
        tmp_slice += tomogram[z, :, :] * kernel[i]
        #tmp_slice += padded_tomogram[z - kernel.size//2, :, :] * kernel[kernel.size // 2]
    #filtered_tomogram[(z - kernel.size//2) % Z_dim, :, :] = tmp_slice
    filtered_tomogram[z, :, :] = tmp_slice
    print(z, end=' ', flush=True)
  print()
  return filtered_tomogram

def filter_over_Y(tomogram, kernel, l, w):
  filtered_tomogram = np.zeros_like(tomogram).astype(np.float32)
  shape_of_tomogram = np.shape(tomogram)
  padded_tomogram = np.zeros(shape=(shape_of_tomogram[0], shape_of_tomogram[1] + kernel.size, shape_of_tomogram[2]))
  padded_tomogram[:, kernel.size//2:shape_of_tomogram[1] + kernel.size//2, :] = tomogram
  Y_dim = tomogram.shape[1]
  for y in range(Y_dim):
    tmp_slice = np.zeros_like(tomogram[:, y, :]).astype(np.float32)
    for i in range(kernel.size):
      if i != kernel.size//2:
        flow = get_flow(padded_tomogram[:, y + i, :], tomogram[:, y, :], l, w)
        OF_compensated_slice = warp_slice(padded_tomogram[:, y + i, :], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
      else:
        # No OF is needed for this slice
        #tmp_slice += tomogram[:, y - kernel.size//2, :] * kernel[kernel.size // 2]
        tmp_slice += tomogram[:, y, :] * kernel[i]
    filtered_tomogram[:, y, :] = tmp_slice
    print(y, end=' ', flush=True)
  print()
  return filtered_tomogram

def filter_over_X(tomogram, kernel, l, w):
  filtered_tomogram = np.zeros_like(tomogram).astype(np.float32)
  shape_of_tomogram = np.shape(tomogram)
  padded_tomogram = np.zeros(shape=(shape_of_tomogram[0], shape_of_tomogram[1], shape_of_tomogram[2] + kernel.size))
  padded_tomogram[:, :, kernel.size//2:shape_of_tomogram[2] + kernel.size//2] = tomogram
  X_dim = tomogram.shape[2]
  for x in range(X_dim):
    tmp_slice = np.zeros_like(tomogram[:, :, x]).astype(np.float32)
    for i in range(kernel.size):
      if i != kernel.size//2:
        flow = get_flow(padded_tomogram[:, :, x + i], tomogram[:, :, x], l, w)
        OF_compensated_slice = warp_slice(padded_tomogram[:, :, x + i], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
      else:
        # No OF is needed for this slice
        #tmp_slice += tomogram[:, :, x - kernel.size//2] * kernel[kernel.size // 2]
        tmp_slice += tomogram[:, :, x] * kernel[i]
    filtered_tomogram[:, :, x] = tmp_slice
    print(x, end=' ', flush=True)
  print()
  return filtered_tomogram

def filter(tomogram, kernel, l, w):
  filtered_tomogram_Z = filter_over_Z(tomogram, kernel, l, w)
  filtered_tomogram_ZY = filter_over_Y(filtered_tomogram_Z, kernel, l, w)
  filtered_tomogram_ZYX = filter_over_X(filtered_tomogram_ZY, kernel, l, w)
  return filtered_tomogram_ZYX
