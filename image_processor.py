import cv2
from random import randint
import numpy as np


class ImageProcessor:

    def __init__(self, background_image_path, overlay_image_path, name='final'):
        self.background_image = cv2.imread(background_image_path)
        self.overlay_image = cv2.imread(overlay_image_path)
        self.name = 'result/' + name
        self.left_bottom = None
        self.left_top = None
        self.right_bottom = None
        self.right_top = None

        self.background_width = None
        self.background_height = None

        self.overlay_width = None
        self.overlay_height = None

        self.x_scale = None
        self.y_scale = None

        self.square_size = 268
        self.line_thickness = 1
        self.line_color = (255, 0, 255)

    def resize_points(self):
        self.x_scale = self.square_size / self.background_width
        self.y_scale = self.square_size / self.background_height

        self.left_bottom = self.rescale_point(self.left_bottom)
        self.left_top = self.rescale_point(self.left_top)
        self.right_bottom = self.rescale_point(self.right_bottom)
        self.right_top = self.rescale_point(self.right_top)

    def rescale_point(self, point):
        return [point[0] * self.x_scale, point[1] * self.y_scale]

    def rand_offset(self):
        interval = int((self.overlay_width + self.overlay_height) / 20)
        return randint(-interval, interval)

    def process(self):
        self.rescale_overlay_image()

        # background center
        cx = self.background_width / 2 + self.rand_offset()
        cy = self.background_height / 2 + self.rand_offset()

        pts1 = np.float32(
            [[0, 0], [self.overlay_width, 0], [0, self.overlay_height], [self.overlay_width, self.overlay_height]])
        self.find_corner_points(cx, cy)
        pts2 = np.float32([self.left_top, self.right_top, self.left_bottom, self.right_bottom])

        h, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        dst = cv2.warpPerspective(self.overlay_image, h, (self.background_width, self.background_height))

        # final = cv2.addWeighted(dst, 0.5, self.background_image, 0.5, 0)

        final = self.get_final_image_using_mask(dst)
        final = cv2.resize(final, (self.square_size, self.square_size))
        cv2.imwrite(self.name + '_0.jpg', final)

        self.resize_points()
        mat = np.zeros((self.square_size, self.square_size, self.background_image.shape[2]), dtype=np.uint8)
        self.draw_line(mat, self.left_top[0], self.left_top[1], self.right_top[0], self.right_top[1])
        self.draw_line(mat, self.right_top[0], self.right_top[1], self.right_bottom[0], self.right_bottom[1])
        self.draw_line(mat, self.right_bottom[0], self.right_bottom[1], self.left_bottom[0], self.left_bottom[1])
        self.draw_line(mat, self.left_bottom[0], self.left_bottom[1], self.left_top[0], self.left_top[1])

        cv2.imwrite(self.name + '_1.jpg', mat)

    def draw_line(self, mat, x1, y1, x2, y2):
        return cv2.line(mat, (int(x1), int(y1)), (int(x2), int(y2)), self.line_color, self.line_thickness)

    def get_final_image_using_mask(self, image):
        mask2 = np.zeros(self.background_image.shape, dtype=np.uint8)
        self.reduce_corner_points()
        roi_corners2 = np.int32([self.left_top, self.right_top, self.right_bottom, self.left_bottom])
        channel_count2 = self.background_image.shape[2]
        ignore_mask_color2 = (255,) * channel_count2
        cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)
        mask2 = cv2.bitwise_not(mask2)
        masked_image2 = cv2.bitwise_and(self.background_image, mask2)
        # Using Bitwise or to merge the two images
        final = cv2.bitwise_or(image, masked_image2)

        return final

    def reduce_corner_points(self):
        i = 2
        self.left_top = [self.left_top[0] + i, self.left_top[1] + i]
        self.right_top = [self.right_top[0] - i, self.right_top[1] + i]
        self.left_bottom = [self.left_bottom[0] + i, self.left_bottom[1] - i]
        self.right_bottom = [self.right_bottom[0] - i, self.right_bottom[1] - i]

    def find_corner_points(self, cx, cy):
        self.left_top = [cx - self.overlay_width / 2 + self.rand_offset(),
                         cy - self.overlay_height / 2 + self.rand_offset()]
        self.right_top = [cx + self.overlay_width / 2 + self.rand_offset(),
                          cy - self.overlay_height / 2 + self.rand_offset()]
        self.left_bottom = [cx - self.overlay_width / 2 + self.rand_offset(),
                            cy + self.overlay_height / 2 + self.rand_offset()]
        self.right_bottom = [cx + self.overlay_width / 2 + self.rand_offset(),
                             cy + self.overlay_height / 2 + self.rand_offset()]

    def rescale_overlay_image(self):
        self.background_height, self.background_width = self.background_image.shape[:2]
        self.overlay_height, self.overlay_width = self.overlay_image.shape[:2]
        overlay_fill_percent = randint(50, 80)
        if self.overlay_height > self.overlay_width:
            scale = ((overlay_fill_percent / 100) * self.background_height) / self.overlay_height
        else:
            scale = ((overlay_fill_percent / 100) * self.background_width) / self.overlay_width
        dim = (int(self.overlay_width * scale), int(self.overlay_height * scale))
        self.overlay_image = cv2.resize(self.overlay_image, dim, interpolation=cv2.INTER_AREA)
        self.overlay_height, self.overlay_width = self.overlay_image.shape[:2]
