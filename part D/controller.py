import numpy
import numpy as np

from SFM import SFM
from sortPoints import SortPoints
from suspicious_points_tfl import Suspicious_points_tfl
from view import View
from PIL import Image
import matplotlib.pyplot as plt

SIDES_SIZE = 40


class Controller:
    def __init__(self, pkl, images):
        self.pkl = pkl
        self.images = images
        self.suspicious_points_tfl = Suspicious_points_tfl()
        self.sfm = SFM(pkl)
        self.sort_points = SortPoints()
        self.view = View()

    def get_images_for_detection(self, image_path, points):
        image = Image.open(image_path)

        # width, height = image.size
        # crop_images = []
        # for point in points:
        #     left = point[0] - 40
        #     top = point[1] + 40
        #     right = point[0] + 40
        #     bottom = point[1] - 40
        #     if point[0] + 40 < width:
        #         right = width
        #     if point[1] + 40 < height:
        #         bottom = height
        #     if point[0] - 40 < 0:
        #         left = 0
        #     if point[1] - 40 < 0:
        #         top = 0
        #     crop_images.append(image.crop((left, top, right, bottom)))
        crop_images = []
        image_mat = np.pad(image, pad_width=((SIDES_SIZE, SIDES_SIZE), (SIDES_SIZE, SIDES_SIZE), (0, 0)),
                           mode='constant', constant_values=0)
        for point in points:
            point = [point[1], point[0]]
            image_croped = image_mat[point[0]:point[0] + 2 * SIDES_SIZE + 1,
                           point[1]:point[1] + 2 * SIDES_SIZE + 1].astype(np.uint8)
            crop_images.append(image_croped)
        return np.array(crop_images)

    def run(self):
        self.suspicious_points_tfl.set_image(self.images[0])
        red_points, green_points = self.suspicious_points_tfl.run()
        merge_points = np.vstack((red_points, green_points))
        merge_points = np.fliplr(merge_points)
        images_for_detection = self.get_images_for_detection(self.images[0], merge_points)
        tfl_bool_prev = self.sort_points.run(images_for_detection)
        tfl_prev = [merge_points[i] for i in range(len(tfl_bool_prev)) if tfl_bool_prev[i] == 1]
        for i in range(1, len(self.images)):
            self.suspicious_points_tfl.set_image(self.images[i])
            red_points, green_points = self.suspicious_points_tfl.run()
            merge_points = np.vstack((red_points, green_points))
            merge_points = np.fliplr(merge_points)
            images_for_detection = self.get_images_for_detection(self.images[i], merge_points)
            tfl_bool_curr = self.sort_points.run(images_for_detection)
            tfl_curr = [merge_points[i] for i in range(len(tfl_bool_curr)) if tfl_bool_curr[i] == 1]
            tfl_curr = numpy.array(tfl_curr)
            self.sfm.set_curr_prev(self.images[i], self.images[i - 1], tfl_curr, tfl_prev)
            curr_container, prev_container, rot_pts, foe = self.sfm.run()
            self.view.set_params(curr_container, prev_container, rot_pts, foe)
            self.view.show_result()
            tfl_prev = tfl_curr
            # fig, (curr_sec) = plt.subplots(1, 1, figsize=(12, 6))
            # curr_sec.imshow(Image.open(self.images[i]))
            # curr_sec.plot(tfl_curr[:, 1], tfl_curr[:, 0], 'b+')
            # plt.show()
