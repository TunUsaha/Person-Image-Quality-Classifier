import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BodyParts:
    """
    Class for storing information about different body parts.
    """
    has_head: bool = False
    has_torso: bool = False
    has_legs: bool = False
    has_feet: bool = False
    head_percentage: float = 0.0
    torso_percentage: float = 0.0
    legs_percentage: float = 0.0
    feet_percentage: float = 0.0


@dataclass
class ImageQualityMetrics:
    """
    Class for storing information about image quality.
    """
    width: int
    height: int
    aspect_ratio: float
    body_parts: BodyParts
    has_light_sources: bool
    good: bool
    reason: str = ""
    has_occlusion: bool = False

class PersonImageAnalyzer:
    def __init__(self, model_path="yolo11n.pt"):
        # ค่ากำหนดต่าง ๆ
        self.MIN_ASPECT_RATIO = 2.0
        self.BRIGHT_THRESHOLD = 240
        self.BRIGHT_PIXEL_PERCENTAGE = 0.01

        # ตำแหน่งจุดสำคัญของร่างกายสำหรับการวิเคราะห์
        # กำหนดจุดสำคัญตามดัชนีของ YOLO keypoints
        # หมายเหตุ: จุดเหล่านี้อาจต้องปรับให้ตรงกับ YOLO v11
        self.HEAD_KEYPOINTS = [0, 1, 2, 3, 4]  # จุดที่เกี่ยวข้องกับใบหน้าและศีรษะ
        self.TORSO_KEYPOINTS = [5, 6, 11, 12]  # ไหล่และสะโพก
        self.LEG_KEYPOINTS = [11, 12, 13, 14, 15, 16]  # สะโพก หัวเข่า ข้อเท้า
        self.FEET_KEYPOINTS = [15, 16, 17, 18, 19, 20]  # ข้อเท้าและเท้า

        # โหลดโมเดล YOLO
        try:
            self.model = YOLO(model_path)
            logger.info(f"โหลดโมเดล YOLO สำเร็จจาก {model_path}")
        except Exception as e:
            logger.error(f"ไม่สามารถโหลดโมเดล YOLO ได้: {str(e)}")
            raise

        # ผลลัพธ์
        self.results = {}

    def analyze_image(self, image_path: str) -> Optional[ImageQualityMetrics]:
        try:
            # อ่านรูปภาพ
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"ไม่สามารถอ่านรูปภาพได้: {image_path}")
                return None

            # คำนวณ aspect ratio
            height, width = image.shape[:2]
            aspect_ratio = height / width

            # ตรวจสอบแหล่งกำเนิดแสง
            has_light_sources = self.detect_light_sources(image)

            # ตรวจจับร่างกายด้วย YOLO
            results = self.model(image, verbose=False)

            # วิเคราะห์ส่วนต่างๆ ของร่างกาย
            body_parts = self.analyze_body_parts(results, height, width)

            # วิเคราะห์การบดบัง
            has_occlusion = self.detect_occlusion(results)

            # เพิ่มการวิเคราะห์คุณภาพภาพ
            image_quality_score, image_quality_reason = self.analyze_image_quality(image)

            # ใช้ Ensemble Decision
            good, reason = self.ensemble_decision(
                ImageQualityMetrics(
                    width=width,
                    height=height,
                    aspect_ratio=aspect_ratio,
                    body_parts=body_parts,
                    has_light_sources=has_light_sources,
                    good=False,  # จะถูกกำหนดค่าใหม่ใน ensemble_decision
                    has_occlusion=has_occlusion
                ),
                image_quality_score,
                image_quality_reason
            )

            # สร้างและส่งคืนผลการวิเคราะห์
            metrics = ImageQualityMetrics(
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                body_parts=body_parts,
                has_light_sources=has_light_sources,
                good=good,
                reason=reason,
                has_occlusion=has_occlusion
            )

            # เก็บผลการวิเคราะห์
            image_filename = os.path.basename(image_path)
            self.results[image_filename] = {
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'has_head': body_parts.has_head,
                'head_percentage': body_parts.head_percentage,
                'has_torso': body_parts.has_torso,
                'torso_percentage': body_parts.torso_percentage,
                'has_legs': body_parts.has_legs,
                'legs_percentage': body_parts.legs_percentage,
                'has_feet': body_parts.has_feet,
                'feet_percentage': body_parts.feet_percentage,
                'has_light_sources': has_light_sources,
                'has_occlusion': has_occlusion,
                'image_quality_score': image_quality_score,
                'good': good,
                'reason': reason
            }

            return metrics

        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการวิเคราะห์รูปภาพ {image_path}: {str(e)}")
            return None

    # ปรับปรุงเมธอด analyze_image_quality ให้มีความละเอียดมากขึ้น
    def analyze_image_quality(self, image):
        """
        วิเคราะห์คุณภาพของภาพโดยใช้หลากหลายเทคนิคที่มีความสมดุล
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # คำนวณความสว่างและความคมชัด
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # ตรวจสอบความชัดของภาพ
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()

        # ให้คะแนน
        score = 100
        reason = ""

        # ตรวจสอบความสว่าง (ปรับให้มีความยืดหยุ่นมากขึ้น)
        if brightness < 40:
            score -= 20
            reason += "too dark, "
        elif brightness > 350:
            score -= 20
            reason += "too bright, "

        # ตรวจสอบความคมชัด
        if contrast < 35:
            score -= 15
            reason += "low contrast, "

        # ตรวจสอบความชัด
        if laplacian < 80:
            score -= 20
            reason += "blurry, "

        # ตัดคำว่า ", " ที่อาจมีอยู่ท้ายสุด
        if reason:
            reason = reason.rstrip(", ")

        return score, reason

    # ปรับเมธอด ensemble_decision ให้มีความสมดุลมากขึ้น
    def ensemble_decision(self, metrics, image_quality_score, image_quality_reason):
        """
        ใช้การตัดสินใจแบบ ensemble ที่มีความสมดุลระหว่าง precision และ recall
        """
        # ตัดสินใจแบบเดิม (ใช้เมธอด check_image_quality จากโค้ดเดิม)
        initial_good, initial_reason = self.check_image_quality(
            metrics.aspect_ratio,
            metrics.body_parts,
            metrics.has_light_sources,
            metrics.has_occlusion
        )

        # สร้างคะแนนสำหรับการตัดสินใจขั้นสุดท้าย
        score = 0
        features = []

        # คำนวณคะแนนจากคุณลักษณะต่างๆ

        # 1. คะแนนจากการตัดสินใจเบื้องต้น
        if initial_good:
            score += 40
            features.append("Passed initial quality check")

        # 2. คะแนนจากคุณภาพภาพ
        if image_quality_score >= 90:
            score += 25
            features.append("Excellent image quality")
        elif image_quality_score >= 85:
            score += 15
            features.append("Good image quality")
        elif image_quality_score < 70:
            score -= 15
            features.append("Poor image quality")

        # 3. คะแนนจากอัตราส่วนภาพ
        if metrics.aspect_ratio >= 2.2:
            score += 20
            features.append("Excellent aspect ratio")
        elif metrics.aspect_ratio >= 1.9 and metrics.aspect_ratio < 2.2:
            score += 10
            features.append("Good aspect ratio")
        elif metrics.aspect_ratio < 1.9:
            score -= 15
            features.append("Poor aspect ratio")

        # 4. คะแนนจากการมองเห็นศีรษะและลำตัว
        if metrics.body_parts.has_head and metrics.body_parts.head_percentage >= 95:
            score += 15
            features.append("Full head visibility")
        elif metrics.body_parts.has_head and metrics.body_parts.head_percentage >= 80:
            score += 8
            features.append("Good head visibility")
        elif not metrics.body_parts.has_head:
            score -= 30
            features.append("No head detected")

        if metrics.body_parts.has_torso and metrics.body_parts.torso_percentage >= 95:
            score += 15
            features.append("Full torso visibility")
        elif metrics.body_parts.has_torso and metrics.body_parts.torso_percentage >= 80:
            score += 8
            features.append("Good torso visibility")
        elif not metrics.body_parts.has_torso:
            score -= 30
            features.append("No torso detected")

        # 5. คะแนนลบจากแหล่งกำเนิดแสงและการบดบัง
        if metrics.has_light_sources:
            score -= 30
            features.append("Light sources detected")

        if metrics.has_occlusion:
            score -= 15
            features.append("Body occlusion detected")

        # ตัดสินใจขั้นสุดท้าย (คะแนนมากกว่าหรือเท่ากับ 40 ถือว่า "good")
        final_good = score >= 60

        # สร้างเหตุผล
        if final_good:
            final_reason = "Image passed with positive features: " + ", ".join(
                [f for f in features if not f.startswith("No ") and
                 not f.startswith("Poor ") and
                 not "detected" in f]
            )
        else:
            final_reason = "Image failed due to: " + ", ".join(
                [f for f in features if f.startswith("No ") or
                 f.startswith("Poor ") or
                 "detected" in f]
            )

        return final_good, final_reason

    def process_image_folder(self, folder_path: str) -> Dict:
        """
        Process all images in the folder.

        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            Dict: Dictionary containing the analysis results.
        """
        logger.info(f"Processing images in folder: {folder_path}")

        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []

        for ext in image_extensions:
            image_files.extend(list(Path(folder_path).glob(f"*{ext}")))

        logger.info(f"Found a total of {len(image_files)} images")

        # Process images one by one
        good_count = 0
        junk_count = 0

        for image_file in image_files:
            logger.info(f"Analyzing: {image_file}")
            metrics = self.analyze_image(str(image_file))

            if metrics:
                if metrics.good:
                    good_count += 1
                else:
                    junk_count += 1
                    logger.info(f"Classified as junk because: {metrics.reason}")

        # Display summary
        total = good_count + junk_count
        if total > 0:  # เพิ่มการตรวจสอบเพื่อป้องกัน division by zero
            logger.info(f"Analysis Summary:")
            logger.info(f"- Total images: {total}")
            logger.info(f"- Good images: {good_count} ({good_count / total * 100:.2f}%)")
            logger.info(f"- Junk images: {junk_count} ({junk_count / total * 100:.2f}%)")
        else:
            logger.warning(f"No images were successfully analyzed in folder: {folder_path}")

        return self.results

    def detect_light_sources(self, image: np.ndarray) -> bool:
        """
        ตรวจสอบแหล่งกำเนิดแสงด้วยวิธีที่ละเอียดขึ้น
        """
        # แปลงเป็น grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # วิธีที่ 1: ตรวจสอบพิกเซลที่สว่างมาก
        bright_pixels = np.sum(gray > self.BRIGHT_THRESHOLD)
        total_pixels = gray.size
        bright_percentage = bright_pixels / total_pixels

        # วิธีที่ 2: ตรวจสอบความแตกต่างระหว่างพื้นที่สว่างที่สุดและมืดที่สุด
        # แบ่งภาพเป็น 10x10 = 100 ส่วน
        h, w = gray.shape
        block_h, block_w = h // 10, w // 10

        max_brightness = 0
        min_brightness = 255

        for i in range(10):
            for j in range(10):
                block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                avg_brightness = np.mean(block)
                max_brightness = max(max_brightness, avg_brightness)
                min_brightness = min(min_brightness, avg_brightness)

        brightness_range = max_brightness - min_brightness

        # ตัดสินใจว่ามีแหล่งกำเนิดแสงหรือไม่
        has_light_sources = bright_percentage > self.BRIGHT_PIXEL_PERCENTAGE or brightness_range > 100

        return has_light_sources

    def analyze_body_parts(self, yolo_results, height: int, width: int) -> BodyParts:
        """
        Analyze different body parts from YOLO results.

        Args:
            yolo_results: Results from YOLO model prediction
            height: Image height
            width: Image width

        Returns:
            BodyParts: Information about body parts visibility and percentage
        """
        body_parts = BodyParts()

        # ตรวจสอบว่ามีผลลัพธ์หรือไม่
        if not yolo_results or len(yolo_results) == 0:
            return body_parts

        # YOLO อาจตรวจพบหลายคนในรูปภาพ เราจะเลือกคนที่มีความเชื่อมั่นสูงสุด
        person_results = None
        max_confidence = -1

        for result in yolo_results:
            # กรองเฉพาะคลาส 'person' (ดัชนี 0 ใน COCO dataset)
            person_boxes = result.boxes[result.boxes.cls == 0]

            if len(person_boxes) > 0:
                # หาคนที่มีความเชื่อมั่นสูงสุด
                confidences = person_boxes.conf.cpu().numpy()
                highest_conf_idx = np.argmax(confidences)

                if confidences[highest_conf_idx] > max_confidence:
                    max_confidence = confidences[highest_conf_idx]
                    person_results = result

        # ถ้าไม่พบคน
        if person_results is None or max_confidence < 0:
            return body_parts

        # ดึงข้อมูล keypoints จากผลลัพธ์
        # YOLO v11 มี keypoints ของร่างกาย (ถ้ามี)
        if hasattr(person_results, 'keypoints') and person_results.keypoints is not None:
            keypoints = person_results.keypoints.data[0].cpu().numpy()  # [x, y, confidence]

            # วิเคราะห์ศีรษะ
            head_visible = False
            head_keypoints = []
            for kp_idx in self.HEAD_KEYPOINTS:
                if kp_idx < len(keypoints) and keypoints[kp_idx][2] > 0.7:  # ความเชื่อมั่น > 0.8
                    head_visible = True
                    head_keypoints.append(keypoints[kp_idx])

            body_parts.has_head = head_visible

            # วิเคราะห์ลำตัว
            torso_visible = False
            torso_keypoints = []
            for kp_idx in self.TORSO_KEYPOINTS:
                if kp_idx < len(keypoints) and keypoints[kp_idx][2] > 0.6:
                    torso_visible = True
                    torso_keypoints.append(keypoints[kp_idx])

            body_parts.has_torso = torso_visible

            # วิเคราะห์ขา
            leg_visible = False
            leg_keypoints = []
            for kp_idx in self.LEG_KEYPOINTS:
                if kp_idx < len(keypoints) and keypoints[kp_idx][2] > 0.3:
                    leg_visible = True
                    leg_keypoints.append(keypoints[kp_idx])

            body_parts.has_legs = leg_visible

            # วิเคราะห์เท้า
            feet_visible = False
            feet_keypoints = []
            for kp_idx in self.FEET_KEYPOINTS:
                if kp_idx < len(keypoints) and keypoints[kp_idx][2] > 0.05:
                    feet_visible = True
                    feet_keypoints.append(keypoints[kp_idx])

            body_parts.has_feet = feet_visible

            # คำนวณระยะห่างเพื่อประมาณความสูงทั้งหมดของร่างกาย
            all_visible_keypoints = []
            for kp in keypoints:
                if kp[2] > 0.5:  # ความเชื่อมั่น > 0.5
                    all_visible_keypoints.append((kp[0], kp[1]))

            # คำนวณความสูงทั้งหมด (จากจุดสูงสุดถึงจุดต่ำสุด)
            if all_visible_keypoints:
                all_y_coords = [p[1] for p in all_visible_keypoints]
                total_height = max(all_y_coords) - min(all_y_coords)
            else:
                total_height = height  # ใช้ความสูงของภาพถ้าไม่มีจุดที่มองเห็น

            # คำนวณเปอร์เซ็นต์ของแต่ละส่วน
            # ศีรษะ
            if body_parts.has_head and len(head_keypoints) > 1:
                head_y_coords = [kp[1] for kp in head_keypoints]
                head_height = max(head_y_coords) - min(head_y_coords)
                body_parts.head_percentage = min(100.0, (head_height / total_height) * 100)

            # ลำตัว
            if body_parts.has_torso and len(torso_keypoints) > 1:
                torso_y_coords = [kp[1] for kp in torso_keypoints]
                torso_height = max(torso_y_coords) - min(torso_y_coords)
                body_parts.torso_percentage = min(100.0, (torso_height / total_height) * 100)

            # ขา
            if body_parts.has_legs and len(leg_keypoints) > 1:
                leg_y_coords = [kp[1] for kp in leg_keypoints]
                leg_height = max(leg_y_coords) - min(leg_y_coords)
                body_parts.legs_percentage = min(100.0, (leg_height / total_height) * 100)

            # เท้า
            if body_parts.has_feet and len(feet_keypoints) > 1:
                feet_y_coords = [kp[1] for kp in feet_keypoints]
                feet_height = max(feet_y_coords) - min(feet_y_coords)
                body_parts.feet_percentage = min(100.0, (feet_height / total_height) * 100)

        # สำหรับกรณีที่ YOLO ไม่มี keypoints ชัดเจน เราอาจใช้กรอบสี่เหลี่ยม (bounding box) แทน
        else:
            # ใช้กรอบสี่เหลี่ยมเพื่อประมาณการณ์ส่วนต่างๆ ของร่างกาย
            person_boxes = person_results.boxes[person_results.boxes.cls == 0]
            if len(person_boxes) > 0:
                # เลือกคนที่มีความเชื่อมั่นสูงสุด
                highest_conf_idx = np.argmax(person_boxes.conf.cpu().numpy())
                box = person_boxes[highest_conf_idx].xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                x1, y1, x2, y2 = box
                box_height = y2 - y1

                # แบ่งกรอบสี่เหลี่ยมเป็นส่วนต่างๆ ตามสัดส่วนโดยประมาณ
                head_height = box_height * 0.15  # ศีรษะประมาณ 15% ของความสูง
                torso_height = box_height * 0.35  # ลำตัวประมาณ 35% ของความสูง
                legs_height = box_height * 0.40  # ขาประมาณ 40% ของความสูง
                feet_height = box_height * 0.10  # เท้าประมาณ 10% ของความสูง

                # กำหนดขอบเขตของแต่ละส่วน
                head_bottom = y1 + head_height
                torso_bottom = head_bottom + torso_height
                legs_bottom = torso_bottom + legs_height

                # ตรวจสอบว่าแต่ละส่วนอยู่ในกรอบภาพหรือไม่
                body_parts.has_head = y1 < head_bottom and y1 >= 0
                body_parts.has_torso = head_bottom < torso_bottom and head_bottom >= 0
                body_parts.has_legs = torso_bottom < legs_bottom and torso_bottom >= 0
                body_parts.has_feet = legs_bottom < y2 and legs_bottom >= 0

                # กำหนดเปอร์เซ็นต์โดยประมาณ
                body_parts.head_percentage = 100.0 if body_parts.has_head else 0.0
                body_parts.torso_percentage = 100.0 if body_parts.has_torso else 0.0
                body_parts.legs_percentage = 100.0 if body_parts.has_legs else 0.0
                body_parts.feet_percentage = 100.0 if body_parts.has_feet else 0.0

        return body_parts

    def check_image_quality(self, aspect_ratio, body_parts, has_light_sources, has_occlusion) -> Tuple[bool, str]:
        """
        เพิ่มความเข้มงวดเพื่อลด False Positive
        """
        score = 100
        reason = ""

        # ตรวจสอบอัตราส่วนภาพ - เพิ่มน้ำหนัก
        if aspect_ratio < self.MIN_ASPECT_RATIO:
            score -= 30
            reason += "Aspect ratio is low, "

        # ตรวจสอบการมองเห็นศีรษะ - เข้มงวดมากขึ้น
        if not body_parts.has_head:
            score -= 60
            reason += "No head detected, "
        elif body_parts.head_percentage < 75:
            score -= 30
            reason += "Head not sufficiently visible, "

        # ตรวจสอบลำตัว - สำคัญ
        if not body_parts.has_torso:
            score -= 60
            reason += "No torso detected, "
        elif body_parts.torso_percentage < 80:
            score -= 30
            reason += "Torso not sufficiently visible, "

        # ตรวจสอบขา
        if not body_parts.has_legs:
            score -= 25
            reason += "No legs detected, "

        # ตรวจสอบแหล่งกำเนิดแสง - เพิ่มบทลงโทษ
        if has_light_sources:
            score -= 25
            reason += "Light sources detected, "

        # ตรวจสอบการบดบัง - เพิ่มบทลงโทษ
        if has_occlusion:
            score -= 25
            reason += "Body occlusion detected, "

        # ปรับเกณฑ์ให้สูงขึ้น
        good = score >= 75

        if reason:
            reason = reason.rstrip(", ")

        return good, reason

    def detect_occlusion(self, yolo_results, visibility_threshold=0.3) -> bool:
        """
        ตรวจสอบว่ามีการบดบังส่วนสำคัญของร่างกายหรือไม่

        Args:
            yolo_results: ผลลัพธ์จาก YOLO
            visibility_threshold: ค่าขีดจำกัดของความมองเห็น

        Returns:
            bool: มีการบดบังหรือไม่
        """
        # ตรวจสอบว่ามีผลลัพธ์หรือไม่
        if not yolo_results or len(yolo_results) == 0:
            return True  # ถ้าไม่เจอคนเลย ถือว่ามีการบดบัง

        # หาผลลัพธ์ของคนที่มีความเชื่อมั่นสูงสุด
        person_results = None
        max_confidence = -1

        for result in yolo_results:
            person_boxes = result.boxes[result.boxes.cls == 0]  # กรองเฉพาะคลาส 'person'

            if len(person_boxes) > 0:
                confidences = person_boxes.conf.cpu().numpy()
                highest_conf_idx = np.argmax(confidences)

                if confidences[highest_conf_idx] > max_confidence:
                    max_confidence = confidences[highest_conf_idx]
                    person_results = result

        # ถ้าไม่พบคน
        if person_results is None or max_confidence < 0:
            return True

        # ตรวจสอบ keypoints ของร่างกาย (ถ้ามี)
        if hasattr(person_results, 'keypoints') and person_results.keypoints is not None:
            keypoints = person_results.keypoints.data[0].cpu().numpy()  # [x, y, visibility]

            # ตรวจสอบจุดสำคัญที่ควรมองเห็นได้ เช่น ศีรษะ ไหล่ และลำตัว
            # ตำแหน่งจุดสำคัญของ YOLO keypoints
            key_landmarks = []
            if len(self.HEAD_KEYPOINTS) > 0:
                key_landmarks.append(self.HEAD_KEYPOINTS[0])  # จุดจมูก/ใบหน้า
            if len(self.TORSO_KEYPOINTS) >= 4:
                key_landmarks.extend(self.TORSO_KEYPOINTS[:4])  # ไหล่และสะโพก

            # ตรวจสอบว่ามีจุดสำคัญที่มองไม่เห็นหรือไม่
            invisible_landmarks = 0
            for idx in key_landmarks:
                if idx < len(keypoints) and keypoints[idx][2] < visibility_threshold:
                    invisible_landmarks += 1

            # ถ้ามีจุดสำคัญที่มองไม่เห็นมากกว่า 1 จุด
            if invisible_landmarks > 1:
                return True

            # ตรวจสอบความไม่สมมาตรของร่างกาย (ซึ่งอาจบ่งชี้ถึงการบดบัง)
            left_side = []
            right_side = []

            # ตรวจสอบไหล่ซ้าย-ขวา
            if len(self.TORSO_KEYPOINTS) >= 2:
                if self.TORSO_KEYPOINTS[0] < len(keypoints):
                    left_side.append(keypoints[self.TORSO_KEYPOINTS[0]][2])  # ไหล่ซ้าย
                if self.TORSO_KEYPOINTS[1] < len(keypoints):
                    right_side.append(keypoints[self.TORSO_KEYPOINTS[1]][2])  # ไหล่ขวา

            # ตรวจสอบสะโพกซ้าย-ขวา
            if len(self.TORSO_KEYPOINTS) >= 4:
                if self.TORSO_KEYPOINTS[2] < len(keypoints):
                    left_side.append(keypoints[self.TORSO_KEYPOINTS[2]][2])  # สะโพกซ้าย
                if self.TORSO_KEYPOINTS[3] < len(keypoints):
                    right_side.append(keypoints[self.TORSO_KEYPOINTS[3]][2])  # สะโพกขวา

            # คำนวณค่าเฉลี่ยความเชื่อมั่นของแต่ละด้าน
            if left_side and right_side:
                left_avg = sum(left_side) / len(left_side)
                right_avg = sum(right_side) / len(right_side)

                # ลดค่าความแตกต่างที่ยอมรับได้
                if abs(left_avg - right_avg) > 0.3:  # ลดจาก 0.4 หรือ 0.6
                    return True

        # ในกรณีที่ YOLO ไม่มีข้อมูล keypoints ให้ใช้การวิเคราะห์จากกรอบสี่เหลี่ยม
        else:
            # ตรวจสอบสัดส่วนของกรอบสี่เหลี่ยม
            # ถ้ากรอบมีความกว้างมากเกินไปเมื่อเทียบกับความสูง อาจบ่งชี้ถึงการมองด้านข้าง (มีการบดบัง)
            person_boxes = person_results.boxes[person_results.boxes.cls == 0]
            if len(person_boxes) > 0:
                highest_conf_idx = np.argmax(person_boxes.conf.cpu().numpy())
                box = person_boxes[highest_conf_idx].xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                # ถ้าความกว้างมากกว่า 70% ของความสูง อาจบ่งชี้ถึงการบดบัง
                if width > height * 0.65:
                    return True

        return False

    def compare_with_original_annotations(self, original_csv: str, new_csv: str, output_dir: str) -> Dict:
        """
        เปรียบเทียบผลการจำแนกกับ annotation เดิม และบันทึกผลการเปรียบเทียบเป็นไฟล์

        Args:
            original_csv (str): พาธไปยังไฟล์ CSV เดิม
            new_csv (str): พาธไปยังไฟล์ CSV ใหม่
            output_dir (str): โฟลเดอร์สำหรับบันทึกผลการเปรียบเทียบ

        Returns:
            Dict: ผลการเปรียบเทียบ
        """
        logger.info(f"กำลังเปรียบเทียบ annotation เดิมและใหม่")

        # อ่านไฟล์ CSV
        try:
            original_df = pd.read_csv(original_csv)
            new_df = pd.read_csv(new_csv)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {str(e)}")
            return {}

        # ตรวจสอบว่ามีคอลัมน์ 'class' ในไฟล์เดิมหรือไม่
        if 'class' not in original_df.columns:
            logger.warning(f"ไม่พบคอลัมน์ 'class' ในไฟล์เดิม ไม่สามารถเปรียบเทียบได้")
            return {}

        # ตรวจสอบว่ามีคอลัมน์ 'class' ในไฟล์ใหม่หรือไม่
        if 'class' not in new_df.columns:
            logger.warning(f"ไม่พบคอลัมน์ 'class' ในไฟล์ใหม่ ไม่สามารถเปรียบเทียบได้")
            return {}

        # เตรียมข้อมูลสำหรับเปรียบเทียบ
        comparison_data = pd.merge(
            original_df[['cropped_filename', 'class']],
            new_df[['cropped_filename', 'class', 'reason']],
            on='cropped_filename',
            suffixes=('_original', '_new')
        )

        # แปลงข้อมูล class เป็น boolean
        comparison_data['original_good'] = comparison_data['class_original'] == 'good'
        comparison_data['new_good'] = comparison_data['class_new'] == 'good'

        # คำนวณความแม่นยำ
        matching = (comparison_data['original_good'] == comparison_data['new_good']).sum()
        total = len(comparison_data)
        accuracy = matching / total * 100 if total > 0 else 0

        # สร้าง confusion matrix
        true_positive = ((comparison_data['original_good'] == True) & (comparison_data['new_good'] == True)).sum()
        true_negative = ((comparison_data['original_good'] == False) & (comparison_data['new_good'] == False)).sum()
        false_positive = ((comparison_data['original_good'] == False) & (comparison_data['new_good'] == True)).sum()
        false_negative = ((comparison_data['original_good'] == True) & (comparison_data['new_good'] == False)).sum()

        # คำนวณค่าสถิติอื่นๆ
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # แสดงผลการเปรียบเทียบ
        logger.info(f"ผลการเปรียบเทียบ:")
        logger.info(f"- จำนวนรูปภาพทั้งหมด: {total}")
        logger.info(f"- จำนวนรูปภาพที่จำแนกตรงกัน: {matching}")
        logger.info(f"- ความแม่นยำ: {accuracy:.2f}%")
        logger.info(f"Confusion Matrix:")
        logger.info(f"- True Positive (original=good, new=good): {true_positive}")
        logger.info(f"- True Negative (original=junk, new=junk): {true_negative}")
        logger.info(f"- False Positive (original=junk, new=good): {false_positive}")
        logger.info(f"- False Negative (original=good, new=junk): {false_negative}")
        logger.info(f"- Precision: {precision:.2f}")
        logger.info(f"- Recall: {recall:.2f}")
        logger.info(f"- F1-score: {f1_score:.2f}")

        # สร้างไฟล์ CSV สำหรับเก็บผลการเปรียบเทียบ
        os.makedirs(output_dir, exist_ok=True)

        # ตั้งชื่อไฟล์ตามกล้อง
        camera_name = os.path.basename(original_csv).split('.')[0]
        comparison_csv = os.path.join(output_dir, f"comparison_{camera_name}.csv")

        # บันทึกข้อมูลการเปรียบเทียบรายภาพ
        comparison_data.to_csv(comparison_csv, index=False)
        logger.info(f"บันทึกข้อมูลการเปรียบเทียบรายภาพไปที่: {comparison_csv}")

        # สร้างไฟล์ CSV สำหรับเก็บสรุปการเปรียบเทียบ
        summary_csv = os.path.join(output_dir, f"summary_{camera_name}.csv")

        # สร้าง DataFrame สำหรับเก็บสรุปผล
        summary_data = pd.DataFrame({
            'camera': [camera_name],
            'total_images': [total],
            'matching_classifications': [matching],
            'accuracy': [accuracy],
            'true_positive': [true_positive],
            'true_negative': [true_negative],
            'false_positive': [false_positive],
            'false_negative': [false_negative],
            'precision': [precision],
            'recall': [recall],
            'f1_score': [f1_score]
        })

        # บันทึกสรุปผลการเปรียบเทียบ
        summary_data.to_csv(summary_csv, index=False)
        logger.info(f"บันทึกสรุปผลการเปรียบเทียบไปที่: {summary_csv}")

        # ส่งคืนผลการเปรียบเทียบ
        comparison_result = {
            'total': total,
            'matching': matching,
            'accuracy': accuracy,
            'true_positive': true_positive,
            'true_negative': true_negative,
            'false_positive': false_positive,
            'false_negative': false_negative,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        return comparison_result

    def set_aspect_ratio_threshold(self, value: float) -> None:
        """
        Set the minimum acceptable aspect ratio.
        """
        self.MIN_ASPECT_RATIO = value
        logger.info(f"Set the minimum aspect ratio to: {value}")

    def update_csv_with_analysis(self, csv_path: str, output_path: str) -> None:
        """
        อัปเดตไฟล์ CSV ด้วยผลการวิเคราะห์

        Args:
            csv_path (str): พาธไปยังไฟล์ CSV ต้นฉบับ
            output_path (str): พาธที่จะบันทึกไฟล์ CSV ใหม่
        """
        logger.info(f"กำลังอัปเดตไฟล์ CSV: {csv_path}")

        # อ่านไฟล์ CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {str(e)}")
            try:
                df = pd.read_csv(csv_path, delimiter='\t')
                logger.info("อ่านไฟล์สำเร็จด้วย tab delimiter")
            except Exception as e2:
                logger.error(f"ไม่สามารถอ่านไฟล์ CSV ได้: {str(e2)}")
                return

        # ทำสำเนาข้อมูลเพื่อไม่ให้กระทบข้อมูลเดิม
        df_copy = df.copy()

        # ตรวจสอบว่ามีคอลัมน์ 'class' อยู่แล้วหรือไม่
        if 'class' in df_copy.columns:
            # เก็บคลาสเดิมไว้ในคอลัมน์ใหม่เพื่อการเปรียบเทียบ
            df_copy['original_class'] = df_copy['class']

        # อัปเดตคอลัมน์ 'class' ด้วยผลการจำแนก
        for index, row in df_copy.iterrows():
            filename = row['cropped_filename']
            if filename in self.results:
                result = self.results[filename]
                df_copy.at[index, 'class'] = 'good' if result['good'] else 'junk'
                # เพิ่มคอลัมน์เหตุผลแยกต่างหาก
                df_copy.at[index, 'reason'] = result['reason']

        # บันทึกไฟล์ใหม่
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_copy.to_csv(output_path, index=False)
        logger.info(f"บันทึกไฟล์ CSV ใหม่ไปที่: {output_path}")

