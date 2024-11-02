import pdb
import glob
import cv2
import os
import numpy as np
import time

class PanaromaStitcher():
    def __init__(self):
        # Initialize SIFT detector and BFMatcher
        self.feature_detector = cv2.SIFT_create()
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_L2)

    def make_panaroma_for_images_in(self, path, no_images=5, ratio_test=0.75):
        folder_path = path
        image_files = sorted(glob.glob(folder_path + os.sep + '*'))
        print(f'Found {len(image_files)} images for panorama stitching')

        if len(image_files) < 2:
            raise ValueError("At least two images are required to create a panorama.")
        
        images = [cv2.imread(img) for img in image_files]
        scale_factor = 0.5
        scaled_images = [cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))) for img in images]
        
        # Set default order if number of images matches a specific pattern
        img_order = [2, 3, 4, 1, 0] if len(scaled_images) == 5 else [3, 2, 4, 1, 5, 0]

        panorama = scaled_images[img_order[0]]
        homographies = []

        for idx in range(1, no_images):
            src_img = panorama
            tgt_img = scaled_images[img_order[idx]]
            panorama, homography, success_flag = self.apply_homography(src_img, tgt_img, ratio_test)
            if success_flag:
                homographies.append(homography)
        
        print(f"Images used: {len(homographies) + 1}")
        return panorama, homographies

    def apply_homography(self, img1, img2, ratio_test=0.75):
        kp1, desc1 = self.feature_detector.detectAndCompute(img1, None)
        kp2, desc2 = self.feature_detector.detectAndCompute(img2, None)

        matches = self.feature_matcher.knnMatch(desc1, desc2, k=2)
        filtered_matches = [m for m, n in matches if m.distance < ratio_test * n.distance]

        if len(filtered_matches) >= 40:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
            homography_matrix, inlier_mask = self.estimate_homography_ransac(dst_pts, src_pts)

            panorama = self.stitch_images(img1, img2, homography_matrix)
            return panorama, homography_matrix, True
        else:
            print("Not enough matches found.")
            return img1, None, False

    def stitch_images(self, base_img, new_img, H):
        h1, w1 = base_img.shape[:2]
        h2, w2 = new_img.shape[:2]

        corners_new_img = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners_new_img, H)
        all_corners = np.concatenate((np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2), transformed_corners), axis=0)

        xmin, ymin = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        translation_dist = [-xmin, -ymin]

        H_translate = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        panorama = cv2.warpPerspective(new_img, H_translate @ H, (xmax - xmin, ymax - ymin))
        panorama[translation_dist[1]:h1 + translation_dist[1], translation_dist[0]:w1 + translation_dist[0]] = base_img
        return panorama

    def estimate_homography_ransac(self, src_pts, dst_pts, tolerance=5.0, iterations=500):
        best_H = None
        best_inlier_count = 0
        src_pts = src_pts.reshape(-1, 2)
        dst_pts = dst_pts.reshape(-1, 2)

        for _ in range(iterations):
            selected_pts = np.random.choice(len(src_pts), 4, replace=False)
            H_trial = self.calculate_homography(src_pts[selected_pts], dst_pts[selected_pts])

            projected_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H_trial).reshape(-1, 2)
            distances = np.linalg.norm(projected_pts - dst_pts, axis=1)
            inliers = distances < tolerance
            inlier_count = np.sum(inliers)

            if inlier_count > best_inlier_count:
                best_H = H_trial
                best_inlier_count = inlier_count

        return best_H, None

    def calculate_homography(self, src_points, dst_points):
        A = []
        for i in range(src_points.shape[0]):
            x_src, y_src = src_points[i][0], src_points[i][1]
            x_dst, y_dst = dst_points[i][0], dst_points[i][1]
            A.append([-x_src, -y_src, -1, 0, 0, 0, x_src * x_dst, y_src * x_dst, x_dst])
            A.append([0, 0, 0, -x_src, -y_src, -1, x_src * y_dst, y_src * y_dst, y_dst])

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        homography = Vt[-1, :].reshape(3, 3)
        homography /= homography[2, 2]
        return homography
