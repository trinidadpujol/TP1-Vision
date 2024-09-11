import cv2
import numpy as np
from matplotlib import pyplot as plt

grayscale = {'cmap':'gray'}

def patch(img, kwargs):

    ret = kwargs.copy()
    if len(img.shape) == 2:
      # num channels == 1
      if 'cmap' not in kwargs:
        ret.update(grayscale)

        # ret["vmax"] = np.max(img)
        # ret["vmin"] = np.min(img)

    return ret

def imshow(
    img,
    **kwargs
  ):

    if len(img.shape) == 3:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    plt.imshow(img, **patch(img, kwargs))
    plt.axis('off')



def show_images(images, titles=None, **kwargs):


  num_images = len(images)
  fig, axs = plt.subplots(1, num_images, figsize=(12, 6))
  if titles is None:
    titles = [None for _ in images]
  for ax, img, title in zip(axs, images, titles):

      if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      ax.imshow(img, **patch(img, kwargs))
      ax.axis('off')
      ax.set_title(title)


def plot_transform(
    r, 
    s, 
    label=None, 
    title=None, 
    fig=None
  ):

  if fig is None:
    plt.figure(figsize=(5, 5))
  if not isinstance(s, list):
    ss = [s]
  else:
    ss = s

  legend = True
  if label is None:
    legend = False
    ls = [None] * len(ss)
  else:
    if not isinstance(label, list):
      ls = [label] * len(ss)
    else:
      ls = label

  for s, lbl in zip(ss, ls):
    plt.plot(r, s, label=lbl)

  plt.grid()
  plt.xlabel("r")
  plt.ylabel("s")
  plt.title(title)
  if legend:
    plt.legend()
  plt.ylim(0, 256)
  plt.xlim(0, 256)

def resize(img, w):
    w_, h_ = img.shape[1], img.shape[0]
    h = int(h_ * w / w_)
    return cv2.resize(img, (w, h))



