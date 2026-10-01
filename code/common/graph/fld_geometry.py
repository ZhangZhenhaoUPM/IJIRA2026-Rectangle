import numpy as np
import cv2

class RectangleDetector:

    def elongate_lines(self, lines):
        extended_lines = []
        extension = (self.elongation_factor - 1.0) / 2.0
        for line in lines:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1
            new_x1 = x1 - dx * extension
            new_y1 = y1 - dy * extension
            new_x2 = x2 + dx * extension
            new_y2 = y2 + dy * extension
            extended_lines.append([new_x1, new_y1, new_x2, new_y2])
        return extended_lines


    def extract_lines(self, gray):
        otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_th2 = max(float(otsu_val), 1.0)
        canny_th1 = max(0.33 * canny_th2, 1.0)
        fld = cv2.ximgproc.createFastLineDetector(length_threshold=15, distance_threshold=1.414, canny_th1=canny_th1, canny_th2=canny_th2, canny_aperture_size=3, do_merge=True)
        lines = fld.detect(gray)
        if lines is None:
            return []
        lines = [line[0].tolist() for line in lines]
        return self.elongate_lines(lines)


    @staticmethod
    def reduce_saturation_brightness(image, saturation_scale=0.5, value_scale=0.5):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * saturation_scale, 0, 255)
        v = np.clip(v * value_scale, 0, 255)
        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

