import numpy as np
import scipy.ndimage
import cv2

def filter_over_Z(tomogram, kernel):
  filtered_tomogram = np.zeros_like(tomogram).astype(np.float32)
  shape_of_tomogram = np.shape(tomogram)
  padded_tomogram = np.zeros(shape=(shape_of_tomogram[0] + kernel.size, shape_of_tomogram[1], shape_of_tomogram[2]))
  padded_tomogram[kernel.size//2:shape_of_tomogram[0] + kernel.size//2, ...] = tomogram
  Z_dim = tomogram.shape[0]
  for z in range(Z_dim):
    tmp_slice = np.zeros_like(tomogram[z, :, :]).astype(np.float32)
    for i in range(kernel.size):
      tmp_slice += padded_tomogram[z + i, :, :] * kernel[i]
    filtered_tomogram[z, :, :] = tmp_slice
    print('.', end='', flush=True)
  print()
  return filtered_tomogram

def filter_over_Y(tomogram, kernel):
  filtered_tomogram = np.zeros_like(tomogram).astype(np.float32)
  shape_of_tomogram = np.shape(tomogram)
  padded_tomogram = np.zeros(shape=(shape_of_tomogram[0], shape_of_tomogram[1] + kernel.size, shape_of_tomogram[2]))
  padded_tomogram[:, kernel.size//2:shape_of_tomogram[1] + kernel.size//2, :] = tomogram
  Y_dim = tomogram.shape[1]
  for y in range(Y_dim):
    tmp_slice = np.zeros_like(tomogram[:, y, :]).astype(np.float32)
    for i in range(kernel.size):
      tmp_slice += padded_tomogram[:, y + i, :] * kernel[i]
    filtered_tomogram[:, y, :] = tmp_slice
    print('.', end='', flush=True)
  print()
  return filtered_tomogram

def filter_over_X(tomogram, kernel):
  filtered_tomogram = np.zeros_like(tomogram).astype(np.float32)
  shape_of_tomogram = np.shape(tomogram)
  padded_tomogram = np.zeros(shape=(shape_of_tomogram[0], shape_of_tomogram[1], shape_of_tomogram[2] + kernel.size))
  padded_tomogram[:, :, kernel.size//2:shape_of_tomogram[2] + kernel.size//2] = tomogram
  X_dim = tomogram.shape[2]
  for x in range(X_dim):
    tmp_slice = np.zeros_like(tomogram[:, :, x]).astype(np.float32)
    for i in range(kernel.size):
      tmp_slice += padded_tomogram[:, :, x + i] * kernel[i]
    filtered_tomogram[:, :, x] = tmp_slice
    print('.', end='', flush=True)
  print()
  return filtered_tomogram

def filter(tomogram, kernel):
  filtered_tomogram_Z = filter_over_Z(tomogram, kernel)
  filtered_tomogram_ZY = filter_over_Y(filtered_tomogram_Z, kernel)
  filtered_tomogram_ZYX = filter_over_X(filtered_tomogram_ZY, kernel)
  return filtered_tomogram_ZYX
