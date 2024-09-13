import cv2
import matplotlib.pyplot as plt
import numpy as np

def match_descriptors(des_A, des_B):
    """ Match descriptors from two images (A and B) using the brute force matcher """
    # Matching descriptors
    bf = cv2.BFMatcher( )
    matches = bf.knnMatch(des_A, des_B, k=2) # returns k best matches for each descriptor in the source image 

    # We want matches whose second best match is much worse than the best match (else, is likely to be noise)
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    return matches, matchesMask

def get_good_matches(matches, matchesMask):
    """ Get the good matches from the matches and matchesMask """
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if matchesMask[i][0] == 1:
            good_matches.append(m)
    return good_matches

def find_homography(kp1, kp2, matches):
    """ Find the homography matrix between two images """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # The reshape is needed to convert the list of points to the shape (n,1,2). n=-1 means the number of points is unknown 
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # 5.0 is the threshold to determine if a point is an inlier or not. The lower the threshold, the more strict the RANSAC algorithm is.
    return H

def warp_images(img1, img2, H, ordered=True):
    """ Warp img1 to img2 using the homography matrix H """
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2) # Coordinates of the 4 corners of the source image
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2) # Coordinates of the 4 corners of the destination image

    pts1_transformed = cv2.perspectiveTransform(pts1, H) # Apply the homography matrix to the source image corners to find the corresponding corners in the destination image

    pts = np.concatenate((pts2, pts1_transformed), axis=0) # Concatenate the corners of the destination image and the corners of the source image warped to the destination image
    
    [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5) # Find the min and max x and y coordinates of the corners of the warped source image in the destination image (subtract 0.5 to round down) 
    [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5) 
    
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]) # Translation matrix 

    img1_warp = cv2.warpPerspective(img1, H_translation.dot(H), (x_max-x_min, y_max-y_min))

    if img1.shape == img2.shape:
        img1_warp[translation_dist[1]:h1 + translation_dist[1], translation_dist[0]:w1 + translation_dist[0]] = img2
        result = img1_warp
    else:
        result = np.zeros((y_max - y_min, x_max - x_min), dtype=img2.dtype)
        result[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]] = img2
        result = np.maximum(result, img1_warp)
    return result

