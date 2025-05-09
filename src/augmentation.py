import cv2
import numpy as np
import random
import inspect
import copy


class Augmentation:

    def __init__(self):
        self.augmentation_methods = [name for name, member in inspect.getmembers(self) if name.startswith('apply')]

    def create_augmented_images(
            self, image: np.ndarray, bboxes: list, num_of_images: int = 3
    ) -> (list, list):
        augmented_images = []
        new_bboxes = []
        for i_id in range(num_of_images):

            augmented_image = copy.deepcopy(image)
            new_bbox = copy.deepcopy(bboxes)
            augmentation_methods = random.sample(self.augmentation_methods, 3)
            augmentation_methods += ['resize_and_paste']
            random.shuffle(augmentation_methods)
            if 'apply_shadow_or_light_leak' in augmentation_methods and 'apply_brightness_contrast' in augmentation_methods:
                augmentation_methods.remove('apply_brightness_contrast')

            for method in augmentation_methods:
                augmented_image, new_bbox = getattr(self, method)(augmented_image, new_bbox)
            augmented_images.append(augmented_image)
            new_bboxes.append(new_bbox)

        return augmented_images, new_bboxes

    @staticmethod
    def apply_compression(
            image: np.ndarray, bboxes: list = None, quality_range: tuple = (35, 45)
    ) -> (np.ndarray, list):
        quality = random.randint(quality_range[0], quality_range[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_image = cv2.imencode('.jpg', image, encode_param)
        compressed_image = cv2.imdecode(encoded_image, 1)
        return compressed_image, bboxes

    @staticmethod
    def apply_shadow_or_light_leak(
            image: np.ndarray,
            bboxes: list = None,
            intensity_shadow_range: tuple = (0.1, 0.2),
            intensity_leak_range: tuple = (0.2, 0.4),
            radius_range: tuple = (0.25, 0.5)
    ) -> (np.ndarray, list):

        h, w = image.shape[:2]

        x = random.randint(0, w)
        y = random.randint(0, h // 2)
        min_dim = min(h, w)
        radius = random.randint(int(min_dim * radius_range[0]), int(min_dim * radius_range[1]))

        effect_type = random.choice(['shadow', 'leak'])

        if effect_type == 'shadow':
            direction = -1
            intensity = random.uniform(intensity_shadow_range[0], intensity_shadow_range[1])
        elif effect_type == 'leak':
            direction = 1
            intensity = random.uniform(intensity_leak_range[0], intensity_leak_range[1])
        else:
            raise ValueError("Invalid effect_type. Choose either 'shadow' or 'leak'.")

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

        mask = np.clip(1 - dist_from_center / radius, 0, 1)
        mask *= direction * intensity
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        mask = np.clip(mask, -1, 1)

        shadow_light_effect = image.astype(np.float32) + (mask[..., np.newaxis] * 255)
        shadow_light_effect = np.clip(shadow_light_effect, 0, 255).astype(np.uint8)

        return shadow_light_effect, bboxes

    @staticmethod
    def light_color_transformation(
            image: np.ndarray, bboxes: list = None, intensity_range: tuple = (0.08, 0.15)
    ) -> (np.ndarray, list):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        color = random.choice(['blue', 'yellow', 'vignette'])
        intensity = random.uniform(intensity_range[0], intensity_range[1])

        if color == 'blue':
            hue_range = (90, 110)
        elif color == 'yellow':
            hue_range = (25, 35)
        else:
            return Augmentation.apply_vignette(image)

        hue_shift = random.randint(hue_range[0], hue_range[1])
        hsv_image[:, :, 0] = hue_shift
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1 - intensity) + (intensity * 100), 0, 255)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * (1 - intensity) + (intensity * 220), 0, 255)

        transformed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return transformed_image, bboxes

    @staticmethod
    def apply_vignette(
            image: np.ndarray, bboxes: list = None, intensity_range: tuple = (0.1, 0.15)
    ) -> (np.ndarray, list):
        h, w = image.shape[:2]

        intensity = random.uniform(intensity_range[0], intensity_range[1])
        kernel_x = cv2.getGaussianKernel(w, int(w * 0.5))
        kernel_y = cv2.getGaussianKernel(h, int(h * 0.5))
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)

        vignette = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            vignette[:, :, i] = image[:, :, i] * (1 - intensity * (1 - mask / 255))

        vignette = np.clip(vignette, 0, 255).astype(np.uint8)
        return vignette, bboxes

    @staticmethod
    def apply_brightness_contrast(
            image: np.ndarray,
            bboxes: list = None,
            brightness_range: tuple = (-30, 30),
            contrast_range: tuple = (-30, 30)
    ) -> (np.ndarray, list):
        brightness = random.uniform(*brightness_range)
        contrast = random.uniform(*contrast_range)

        image = image.astype(np.float32)
        image = image * (contrast / 127 + 1) - contrast + brightness
        image = np.clip(image, 0, 255)

        return image.astype(np.uint8), bboxes

    @staticmethod
    def apply_lighting_effects(
            image: np.ndarray, bboxes: list = None, intensity_range: tuple = (0.5, 0.7)
    ) -> (np.ndarray, list):
        rows, cols = image.shape[:2]

        intensity = random.uniform(intensity_range[0], intensity_range[1])
        direction = random.choice(['left-to-right', 'right-to-left', 'top-to-bottom', 'bottom-to-top'])

        if direction == 'left-to-right':
            gradient = np.tile(np.linspace(intensity, 1.0, cols), (rows, 1))
        elif direction == 'right-to-left':
            gradient = np.tile(np.linspace(1.0, intensity, cols), (rows, 1))
        elif direction == 'top-to-bottom':
            gradient = np.tile(np.linspace(intensity, 1.0, rows), (cols, 1)).T
        elif direction == 'bottom-to-top':
            gradient = np.tile(np.linspace(1.0, intensity, rows), (cols, 1)).T
        else:
            raise ValueError(
                "Invalid direction. Choose from 'left-to-right', 'right-to-left', 'top-to-bottom', 'bottom-to-top'.")

        gradient = np.dstack([gradient] * 3)
        lighting_img = np.clip(image[:, :, :3] * gradient, 0, 255).astype(np.uint8)

        if image.shape[2] == 4:
            alpha_channel = image[:, :, 3]
            lighting_img = np.dstack([lighting_img, alpha_channel])

        return lighting_img, bboxes

    @staticmethod
    def rotate_bbox(bbox, angle, image_shape):
        H, W = image_shape[:2]
        cx, cy = W / 2, H / 2

        x_min, y_min, x_max, y_max = bbox
        points = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])

        theta = np.radians(angle)

        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        rotated_points = np.dot(points - np.array([cx, cy]), rotation_matrix.T) + np.array([cx, cy])

        x_min_new, y_min_new = np.min(rotated_points, axis=0)
        x_max_new, y_max_new = np.max(rotated_points, axis=0)

        return int(x_min_new), int(y_min_new), int(x_max_new), int(y_max_new)

    @staticmethod
    def apply_rotation(
            image: np.ndarray, bboxes: list = None, angle_range: tuple = (-0.5, 0.5)
    ) -> (np.ndarray, list):
        angle = random.uniform(angle_range[0], angle_range[1])
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))

        return rotated, bboxes

    @staticmethod
    def resize_and_paste(
            image: np.ndarray, bboxes: list = None, scale_range: tuple = (0.7, 0.95)
    ) -> (np.ndarray, list):
        (h, w) = image.shape[:2]
        scale = random.uniform(scale_range[0], scale_range[1])
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        if len(image.shape) == 3:
            channels = image.shape[2]
        else:
            channels = 1
        background = np.full((h, w, channels), 255, dtype=image.dtype)

        max_x = w - new_w
        max_y = h - new_h
        offset_x = random.randint(0, max_x)
        offset_y = random.randint(0, max_y)

        background[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        if bboxes is None:
            return background

        if len(bboxes) > 2 and isinstance(bboxes[0][0], float):
            bboxes = [bboxes]

        new_bboxes_tuples = []
        for i, bboxes_tuple in enumerate(bboxes):
            new_bboxes = []
            for bbox in bboxes_tuple:
                if i == 1:
                    coords = bbox
                    label = None
                else:
                    label, coords = bbox[0], bbox[1:]
                new_coords = [
                    coords[0] * scale + offset_x,
                    coords[1] * scale + offset_y,
                    coords[2] * scale + offset_x,
                    coords[3] * scale + offset_y
                ]
                if label is not None:
                    new_bboxes.append([label] + new_coords)
                else:
                    new_bboxes.append(new_coords)
            new_bboxes_tuples.append(new_bboxes)

        if len(new_bboxes_tuples) == 1:
            return background, new_bboxes_tuples[0]
        return background, new_bboxes_tuples

    @staticmethod
    def apply_lines(image: np.ndarray, bboxes: list = None) -> (np.ndarray, list):
        augmented = image.copy()
        h, w = augmented.shape[:2]

        num_lines = random.randint(2, 6)

        for i in range(num_lines):
            cx = random.randint(w // 8, 7 * w // 8)
            cy = random.randint(0, h // 2)

            length = random.randint(w // 4, w // 2)

            angle_deg = random.uniform(-45, 45) if i % 2 == 0 else 0
            angle_rad = np.deg2rad(angle_deg)

            dx = int((length / 2) * np.cos(angle_rad))
            dy = int((length / 2) * np.sin(angle_rad))
            pt1 = (cx - dx, cy - dy)
            pt2 = (cx + dx, cy + dy)

            color_val = random.randint(0, 80)
            color = (color_val, color_val, color_val) if len(augmented.shape) == 3 else color_val

            thickness = random.randint(1, 3)

            cv2.line(augmented, pt1, pt2, color, thickness)

        return augmented, bboxes
