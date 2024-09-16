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

def compute_homography(src_points, dst_points):
    """Compute homography matrix from source and destination points."""

    A = []
    for i in range(src_points.shape[0]):
        x_src, y_src = src_points[i, 0, 0], src_points[i, 0, 1]
        x_dst, y_dst = dst_points[i, 0, 0], dst_points[i, 0, 1]
        A.extend([
            [-x_src, -y_src, -1, 0, 0, 0, x_dst * x_src, x_dst * y_src, x_dst],
            [0, 0, 0, -x_src, -y_src, -1, y_dst * x_src, y_dst * y_src, y_dst]
        ])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)

    return H

def ransac_homography(kp1, kp2, matches, t=5, max_iterations=1000):
    """Estimate homography using RANSAC."""
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    best_inliers = []
    num_points = src_pts.shape[0]

    for _ in range(max_iterations):
        # Random sample of 4 points
        indices = np.random.choice(num_points, 4, replace=False)
        sample_src = src_pts[indices]
        sample_dst = dst_pts[indices]

        # Compute homography from the sample
        H = compute_homography(sample_src, sample_dst)

        # Calculate the number of inliers based on the reprojection error and tolerance t
        inliers = []
        for i in range(num_points):
            # Map source point to destination using H
            src_pt = np.append(src_pts[i, 0], 1)  # Accessing the 2D point
            estimated_pt = np.dot(H, src_pt)
            estimated_pt /= estimated_pt[2]  # Normalize by the third coordinate

            # Calculate the error as the Euclidean distance
            error = np.linalg.norm(dst_pts[i, 0] - estimated_pt[:2])

            # Count as inlier if the error is below the threshold
            if error < t:
                inliers.append(i)

        # Update the best_inliers list if current model is better
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    # Final homography with best inliers and minimum squares method
    H, _ = cv2.findHomography(src_pts[best_inliers], dst_pts[best_inliers], method=cv2.LMEDS)

    return H, best_inliers

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

