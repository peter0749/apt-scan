import numpy as np
import math
from skimage.measure import regionprops
from skimage.measure import label

def find_corner_condidate(binary_img, component_threshold=13):
    h, w = binary_img.shape[:2]
    cx = w // 2
    cy = h // 2
    l = label(binary_img)
    p = regionprops(l, cache=True)
    num = len(p)
    if num==0:
        return np.array([])
    points = []
    dists = []
    for region in p:
        point = region.centroid
        area  = region.area
        if area < component_threshold:
            continue
        y, x = point
        points.append(point)
        dists.append( [area, (y-cy)**2 + (x-cx)**2] ) # (-area of reagion, distance between center of image and region center)
    if len(points)==0:
        return np.array([])
    dists = np.asarray(dists)
    # print(dists)
    points = np.asarray(points)
    # print(points)
    order = np.lexsort((dists[...,1], -dists[...,0]))
    # print(order)
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
    points = points[order] # clockwise order (sort by atan2)
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

