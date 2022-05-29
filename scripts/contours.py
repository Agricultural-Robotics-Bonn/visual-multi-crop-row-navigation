
import cv2 as cv
import numpy as np
from geometric import *

def getContourCenter(contours):
        """function to compute the center points of the contours

        Args:
            contours (_type_): contours from image

        Returns:
            _type_: contours centers
        """
        # get center of contours
        contCenterPTS = np.zeros((len(contours),2))
        for i in range(0, len(contours)):
            # get contour
            c_curr = contours[i];
            
            # get moments
            M = cv.moments(c_curr)
            
            # compute center of mass
            if(M['m00'] != 0):
               cx = int(M['m10']/M['m00'])
               cy = int(M['m01']/M['m00'])
               contCenterPTS[i,:] = [cy,cx]

        contCenterPTS = contCenterPTS[~np.all(contCenterPTS == 0, axis=1)]
        return contCenterPTS

def getPlantMasks(binrayMask, min_contour_area, bushy=False):
        # find contours
        contours = cv.findContours(binrayMask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[1]

        # cv.imshow("crop rows mask",self.mask)
        # self.handleKey()

        # filter contours based on size
        # self.filtered_contours = contours
        filtered_contours = list()
        for i in range(len(contours)):
            if cv.contourArea(contours[i]) > min_contour_area:
                
                if bushy : 
                    cn_x, cn_y, cnt_w, cn_h = cv.boundingRect(contours[i])

                    # split to N contours h/max_coutour_height
                    sub_contours = splitContours(contours[i], cn_x, cn_y, cnt_w, cn_h)
                    for j in sub_contours:
                        if j != []:
                            filtered_contours.append(j)
                else:
                    if contours[i] != []:
                        filtered_contours.append(contours[i]) 
            # else:
            #     filtered_contours.append(contours[i]) 
        return filtered_contours
    
def splitContours(contour, x, y, w, h, max_coutour_height):
    """splits larg contours in smaller regions 

    Args:
        contour (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        w (_type_): _description_
        h (_type_): _description_

    Returns:
        _type_: sub polygons (seperated countours)
    """
    sub_polygon_num = h // max_coutour_height
    sub_polys = list()
    subContour = list()
    vtx_idx = list()
    contour = [contour.squeeze().tolist()]
    for subPoly in range(1, sub_polygon_num + 1):
        for vtx in range(len(contour[0])): 
            if  (subPoly - 1 * max_coutour_height) -1 <=  contour[0][vtx][1] and \
                (subPoly * max_coutour_height) -1 >= contour[0][vtx][1] and \
                vtx not in vtx_idx:
                subContour.append([contour[0][vtx]])
                vtx_idx.append(vtx)

        sub_polys.append(np.array(subContour))
        subContour = list()

    return sub_polys

def sortContours(contours, method="left-to-right"):
    """initialize the reverse flag and sort index

    Args:
        cnts (_type_): _description_
        method (str, optional): _description_. Defaults to "left-to-right".

    Returns:
        _type_: sorted countours, bboxes
    """
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return contours, boundingBoxes

def getContoursInWindow(contourCenters, box):
    """iflters out countours inside a box

    Args:
        contourCenters (_type_): _description_
        box (_type_): _description_

    Returns:
        _type_: contour centers
    """
    points = []
    for cnt in range(len(contourCenters[1])):
        x, y = contourCenters[0][cnt], contourCenters[1][cnt]
        if isInBox(list(box), [x, y]):
            points.append([x, y])
    return points
