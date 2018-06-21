import numpy as np
import math
from skimage.measure import regionprops
from skimage.measure import label

def find_corner_condidate(binary_img):
    h, w = binary_img.shape[:2]
    cx = w // 2
    cy = h // 2
    l = label(binary_img)
    p = regionprops(l, cache=True)
    num = len(p)
    points = []
    dists = []
    for region in p:
        point = region.centroid
        y, x = point
        points.append(point)
        dists.append( (y-cy)**2 + (x-cx)**2 ) # amp
    dists = np.asarray(dists)
    points= np.asarray(points)
    # print(points)
    order = np.argsort(dists)
    del dists
    points = points[order][:4]
    # print(points)
    angles = []
    for (y,x) in points:
        vy, vx = (y-cy), (x-cx) # vector
        angles.append(math.atan2(vy, vx))
    angles = np.asarray(angles)
    order = np.argsort(angles)
    del angles
    points = points[order] # clockwise order
    return points

if __name__ == '__main__':
    from skimage.io import imread
    from skimage.draw import circle
    import matplotlib.pyplot as plt
    points = np.array([
                        [10, 20 ], # lu
                        [20, 100], # ru
                        [100,100], # rd
                        [100, 20], # ld
                      ])
    test_img = np.zeros((192,256), dtype=np.uint8)
    for p in points:
        rr, cc = circle(*p, 5, shape=(192,256))
        test_img[rr, cc] = 1

    plt.imshow(test_img)
    plt.show()
    print('Expected corners:')
    for p in points:
        print(p)
    points = find_corner_condidate(test_img)
    print('Corners found:')
    for p in points:
        print(p)

