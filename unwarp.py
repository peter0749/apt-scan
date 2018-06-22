import postprocess
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def order_points(pts): # format: (x, y)
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts[...,0]**2 + pts[...,1]**2
    
    lr = np.argmin(s)
    rd = np.argmax(s)
    
    rect[0] = pts[lr]
    rect[2] = pts[rd]
    
    mask = np.ones(4, dtype=np.bool)
    mask[lr] = False
    mask[rd] = False
    
    pts = pts[mask]
    
    rect[1] = pts[np.argmax(pts[...,0])] # upper-right
    rect[3] = pts[np.argmin(pts[...,0])] # lower-left

    # return the ordered coordinates
    return rect
def three_pts_to_four_pts(pts): # (3, 2)
    apts = np.append(pts[-1:], pts, axis=0) # (4, 2) 
    bpts = np.append(pts, pts[0:1], axis=0) # (4, 2)
    a2b = apts-bpts # (4, 2)
    dists = np.sqrt(np.sum((a2b)**2, axis=-1)) # (4,)
    max_dist = np.argmax(dists) 
    a = max_dist-1
    b = max_dist
    mask = np.ones(3,dtype=np.bool)
    mask[[a,b]] = False
    c = np.squeeze(pts[mask]) # 1 point
    a, b = pts[a], pts[b]
    m = 0.5 * (a+b)
    v = m - c
    d = m + v
    return np.array([a, c, b, d], dtype=np.float32)

def robust_unwarp(image, pts):
    if len(pts)<3: # too less points
        return image # give up!
    elif len(pts)==3: # hmmmmmmmmm
        pts = three_pts_to_four_pts(pts)
        return unwarp(image, pts) # no problem
    else:
        return unwarp(image, pts) # no problem

def unwarp(image, pts): # format: (y,x)
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts[...,::-1]) # format: (x,y)
    # print(rect)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    aveWidth  = int((widthA+widthB)//2) # max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    aveHeight = int((heightA+heightB)//2) # max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [aveWidth - 1, 0],
        [aveWidth - 1, aveHeight - 1],
        [0, aveHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (aveWidth, aveHeight))
    
    # M = cv2.getAffineTransform(rect[:3], dst[:3])
    # warped = cv2.warpAffine(image, M, (aveWidth, aveHeight))

    # return the warped image
    return warped

if __name__ == '__main__':
    from skimage.io import imread
    from skimage.draw import circle
    import matplotlib.pyplot as plt
    points = np.array([
                        [10, 20 ], # lu
                        # [20, 100], # ru
                        [100,120], # rd
                        [100, 20], # ld
                      ])
    test_img = np.zeros((192,128), dtype=np.uint8)
    for p in points:
        rr, cc = circle(*p, 5, shape=(192,128))
        test_img[rr, cc] = 1
    points = postprocess.find_corner_condidate(test_img)
    img = Image.fromarray(test_img*255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("assets/calibri.ttf", 18)
    draw.text((50, 50),"hi",255,font=font)
    test_img = np.array(img)
    w = robust_unwarp(test_img, points)
    plt.imshow(test_img)
    plt.show()
    plt.imshow(w)
    plt.show()
