import os
import random
import numpy as np
import torch
import visdom


#initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def adjust_learning_rate(optimizer, epoch, phase):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if (epoch%phase==(phase-1)):
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr']/10.

def display_result(pts,config,pts_per_prim,epoch,it,vis,mode):
    #display everything on visdom

    #create the points labels
    #---------------------------------------------------------------------------
    label = 1
    labels_generated_pts = torch.Tensor()
    points_repartition = pts_per_prim[0]

    for i in range(points_repartition.size(0)):
        if(points_repartition[i] != 0):
            ones  = torch.ones(points_repartition[i].data[0])*label
            label = label + 1
            labels_generated_pts = torch.cat((labels_generated_pts,ones),0)
    #---------------------------------------------------------------------------

    #display the training input/output
    #---------------------------------------------------------------------------
    vis.scatter(X = pts.transpose(2,1).contiguous()[0].data.cpu(),
                win = mode+' set input',
                opts = dict(title = mode+" set input", markersize = 2,),)

    x = []
    for i in range(pts_per_prim.size(1)):
        n = torch.ones(points_repartition[i].data[0])*i
        x.append(n)
    x = torch.cat(x,0)


    vis.histogram(X = x,
                  win = "repartition " + mode,
                  opts = dict(title = "repartition " + mode, markersize = 2,),)

    vis.scatter(X = config[0].data.cpu(),
                Y = labels_generated_pts,
                win = mode+' set ouput',
                opts = dict(title=mode+" set output",markersize=2,),)
    #---------------------------------------------------------------------------


    #display some primitives
    #---------------------------------------------------------------------------
    X       = config[0].data.cpu()
    run     = True
    current = 0
    i       = 0
    j       = 0

    while(run):

        points = pts_per_prim[0,i].data[0]
        if(points !=0):
            title = mode+" primitive "+str(j+1)
            xX  = X[current:current+points]
            opt = dict(title=title, markersize=2)
            vis.scatter(xX,win = title,opts = opt)
            j = j+1
            current=current+points
        i = i+1

        if j ==2:
            run = False
    #---------------------------------------------------------------------------


def generate_cude(h,l,w):
    return 0


def adjust(pts_per_prim,num_points):
# make sure that the number of predicted points is equal in all the batch

    #compute the number of points to add or substract per batch
    #---------------------------------------------------------------------------
    sum_batch = torch.sum(pts_per_prim,1)
    diff = sum_batch - num_points
    #---------------------------------------------------------------------------

    #add or substract points to the biggest primitive
    #---------------------------------------------------------------------------
    m, index = torch.max(pts_per_prim,1)
    for i in range(diff.size(0)):
        new_val = pts_per_prim[i,index[i]] - diff[i]
        pts_per_prim[i,index[i]] = new_val
    #---------------------------------------------------------------------------

    return pts_per_prim


class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


CHUNK_SIZE = 150
lenght_line = 60
def my_get_n_random_lines(path, n=5):
    MY_CHUNK_SIZE = lenght_line * (n+2)
    lenght = os.stat(path).st_size
    with open(path, 'r') as file:
            file.seek(random.randint(400, lenght - MY_CHUNK_SIZE))
            chunk = file.read(MY_CHUNK_SIZE)
            lines = chunk.split(os.linesep)
            return lines[1:n+1]

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

#Example:
def get_colors(num_colors=10):
  colors = []
  for i in range(0,num_colors):
      colors.append(generate_new_color(colors,pastel_factor = 0.9))
  for i in range(0,num_colors):
      for j in range(0,3):
        colors[i][j] = int(colors[i][j]*256)
      colors[i].append(255)
  return colors


#CODE from 3D R2N2
def image_transform(img, crop_x, crop_y, crop_loc=None, color_tint=None):
    """
    Takes numpy.array img
    """

    # Slight translation
    if not crop_loc:
        crop_loc = [np.random.randint(0, crop_y), np.random.randint(0, crop_x)]

    if crop_loc:
        cr, cc = crop_loc
        height, width, _ = img.shape
        img_h = height - crop_y
        img_w = width - crop_x
        img = img[cr:cr + img_h, cc:cc + img_w]
        # depth = depth[cr:cr+img_h, cc:cc+img_w]

    if np.random.rand() > 0.5:
        img = img[:, ::-1, ...]

    return img


def crop_center(im, new_height, new_width):
    height = im.shape[0]  # Get dimensions
    width = im.shape[1]
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return im[top:bottom, left:right]


def add_random_color_background(im, color_range):
    r, g, b = [np.random.randint(color_range[i][0], color_range[i][1] + 1) for i in range(3)]
    if isinstance(im, Image.Image):
        im = np.array(im)

    if im.shape[2] > 3:
        # If the image has the alpha channel, add the background
        alpha = (np.expand_dims(im[:, :, 3], axis=2) == 0).astype(np.float)
        im = im[:, :, :3]
        bg_color = np.array([[[r, g, b]]])
        im = alpha * bg_color + (1 - alpha) * im

    return im


def preprocess_img(im, train=True):
    # add random background
    # im = add_random_color_background(im, cfg.TRAIN.NO_BG_COLOR_RANGE if train else
                                     # cfg.TEST.NO_BG_COLOR_RANGE)

    # If the image has alpha channel, remove it.
    im_rgb = np.array(im)[:, :, :3].astype(np.float32)
    if train:
        t_im = image_transform(im_rgb, 17, 17)
    else:
        t_im = crop_center(im_rgb, 224, 224)

    # Scale image
    t_im = t_im / 255.

    return t_im


if __name__ == '__main__':

  #To make your color choice reproducible, uncomment the following line:
  #random.seed(10)

  colors = get_colors(10)
  print "Your colors:",colors
