import sewar.full_ref as metrics_cal
from skimage.exposure import adjust_gamma, rescale_intensity
import h5py as h5
import numpy as np
import sys

import matplotlib.pyplot as plt

import os

def rescale(reconstruction, hi):
  I = rescale_intensity(reconstruction, out_range=(0., 1.))
  return adjust_gamma(I, 1, hi)

def iqcheck(f1, f2):
  hi = 1

  # read hdf5 file and return numpy array
  with h5.File(f1, 'r') as f:
    recon0 = np.array(f['data'])

  with h5.File(f2, 'r') as f:
    recon1 = np.array(f['data'])

  recon0 = rescale(np.rot90(recon0[0])[::-1], hi)
  recon1 = rescale(np.rot90(recon1[0])[::-1], hi)

  # readjust the range of the image to 0-255
  recon0i = (recon0 * 255).astype(np.uint8)
  recon1i = (recon1 * 255).astype(np.uint8)

  q0 = metrics_cal.msssim(recon1i, recon0i)
  q1 = metrics_cal.ssim(recon1i, recon0i)
  q2 = metrics_cal.uqi(recon1, recon0)
  q3 = metrics_cal.mse(recon1, recon0)
  q4 = metrics_cal.psnr(recon1i, recon0i)

  # print("MS-SSIM: {}; SSIM: {}; UQI: {}; MSE: {}; PSNR: {}".format(q0, q1, q2, q3, q4))
  return q0, q1, q2, q3, q4

def plot_quality(data, name, figpath):
  plt.figure()
  plt.plot(data)
  plt.xlabel("# outer iterations")
  plt.ylabel(name)

  plt.tight_layout()
  plt.savefig(figpath)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python quality-time.py <recon folder>")
    sys.exit(1)

  # Collect recon files
  fdir = sys.argv[1]
  files = []
  for file in os.listdir(fdir):
    if file.endswith(".h5"):
      files.append(os.path.join(fdir, file))
  files = sorted(files)

  fbase = files[0]
  files = files[1:20]

  msssims = []
  ssims = []
  uqis = []
  mses = []
  psnrs = []

  for file in files:
    print("checking quality for", file)
    msssim, ssim, uqi, mse, psnr = iqcheck(fbase, file)
    msssims.append(msssim)
    ssims.append(ssim)
    uqis.append(uqi)
    mses.append(mse)
    psnrs.append(psnr)
    fbase = file

  figpath = "figures/"
  plot_quality(msssims, "MS-SSIM", figpath + "msssim")
  plot_quality(ssims, "SSIM", figpath + "ssim")
  plot_quality(uqis, "UQI", figpath + "uqi")
  plot_quality(mses, "MSE", figpath + "mse")
  plot_quality(psnrs, "PSNR", figpath + "psnr")




 