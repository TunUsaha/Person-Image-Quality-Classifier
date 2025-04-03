#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import the modules we created
from data_preparation import DataPreparation
from image_analyzer import PersonImageAnalyzer

# Set logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"image_analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse arguments from the command line
    """
    parser = argparse.ArgumentParser(description='Analyze person images and filter based on quality criteria')

    parser.add_argument('--csv1', type=str,
                        default="/Users/ahasunut/PycharmProjects/PythonProject/YOLOv11_con70_person_sequence_cam1/image_annotations_cropped_cam1.csv",
                        help='Path to the CSV file of camera 1')

    parser.add_argument('--csv2', type=str,
                        default="/Users/ahasunut/PycharmProjects/PythonProject/YOLOv11_con70_person_sequence_cam2/image_annotations_cropped_cam2.csv",
                        help='Path to the CSV file of camera 2')

    parser.add_argument('--images1', type=str,
                        default="/Users/ahasunut/PycharmProjects/PythonProject/YOLOv11_con70_person_sequence_cam1/person_images_cam1",
                        help='Path to the folder of images from camera 1')

    parser.add_argument('--images2', type=str,
                        default="/Users/ahasunut/PycharmProjects/PythonProject/YOLOv11_con70_person_sequence_cam2/person_images_cam2",
                        help='Path to the folder of images from camera 2')

    parser.add_argument('--output', type=str, default="output",
                        help='Path to save the results')

    parser.add_argument('--min-aspect-ratio', type=float, default=1.5,
                        help='Minimum acceptable aspect ratio value')

    parser.add_argument('--yolo-model', type=str, default="yolo11n.pt",
                        help='Path to the YOLO model file')

    parser.add_argument('--skip-preparation', action='store_true',
                        help='Skip the data preparation step')

    parser.add_argument('--only-preparation', action='store_true',
                        help='Perform only the data preparation step')

    parser.add_argument('--visualize', action='store_true',
                        help='Generate graphs and visualizations')

    return parser.parse_args()


def prepare_data(args):
    """
    Prepare data from CSV files and images
    """
    logger.info("Starting data preparation steps...")

    # Create a folder for results
    output_dir = Path(args.output)
    output_csv_dir = output_dir / "csv"
    os.makedirs(output_csv_dir, exist_ok=True)

    # Define paths for output files
    prepared_csv_cam1 = output_csv_dir / "prepared_cam1.csv"
    prepared_csv_cam2 = output_csv_dir / "prepared_cam2.csv"
    prepared_csv_combined = output_csv_dir / "prepared_combined.csv"

    # Create an instance of DataPreparation
    data_prep = DataPreparation(
        args.csv1,
        args.csv2,
        args.images1,
        args.images2
    )

    # Load CSV data
    logger.info("Loading data from CSV files...")
    data_cam1, data_cam2 = data_prep.load_csv_data()

    # Check image files
    logger.info("Checking image files...")
    missing_images = data_prep.check_image_files()

    # Add aspect ratio
    logger.info("Calculating aspect ratio...")
    data_cam1, data_cam2 = data_prep.add_aspect_ratio_column()

    # Analyze aspect ratio distribution
    logger.info("Analyzing aspect ratio distribution...")
    stats_cam1, stats_cam2 = data_prep.analyze_aspect_ratio_distribution()

    # Generate visualizations for the distribution (if requested)
    if args.visualize:
        visualize_aspect_ratio_distribution(data_cam1, data_cam2, output_dir)

    # Save the prepared data
    logger.info("Saving the prepared data...")
    data_prep.save_prepared_data(prepared_csv_cam1, prepared_csv_cam2, prepared_csv_combined)

    logger.info("Data preparation steps completed successfully")

    return prepared_csv_cam1, prepared_csv_cam2, prepared_csv_combined


def analyze_images(args, prepared_csv_cam1, prepared_csv_cam2):
    """
    Analyze images and filter them based on quality criteria
    """
    logger.info("Starting image analysis steps...")

    # Create folders for results
    output_dir = Path(args.output)
    output_csv_dir = output_dir / "csv"
    output_comparison_dir = output_dir / "comparison"
    output_vis_dir = output_dir / "visualizations"

    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_comparison_dir, exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)

    # Define paths for output files
    analyzed_csv_cam1 = output_csv_dir / "analyzed_cam1.csv"
    analyzed_csv_cam2 = output_csv_dir / "analyzed_cam2.csv"

    # Create an instance of PersonImageAnalyzer with the specified YOLO model
    analyzer = PersonImageAnalyzer(model_path=args.yolo_model)
    logger.info(f"Initialized PersonImageAnalyzer with YOLO model: {args.yolo_model}")

    # Set the aspect ratio threshold
    analyzer.MIN_ASPECT_RATIO = args.min_aspect_ratio
    logger.info(f"Set minimum aspect ratio to: {args.min_aspect_ratio}")

    # Process images for both cameras
    logger.info("Analyzing images from camera 1...")
    results_cam1 = analyzer.process_image_folder(args.images1)

    logger.info("Analyzing images from camera 2...")
    results_cam2 = analyzer.process_image_folder(args.images2)

    # Update CSV files with analysis results
    logger.info("Updating CSV files with analysis results...")
    analyzer.update_csv_with_analysis(prepared_csv_cam1, analyzed_csv_cam1)
    analyzer.update_csv_with_analysis(prepared_csv_cam2, analyzed_csv_cam2)

    # Compare with original annotations (if classes are already provided)
    logger.info("Comparing with original annotations...")
    comparison_cam1 = analyzer.compare_with_original_annotations(args.csv1, analyzed_csv_cam1, output_comparison_dir)
    comparison_cam2 = analyzer.compare_with_original_annotations(args.csv2, analyzed_csv_cam2, output_comparison_dir)

    # Generate visualizations for the analysis results (if requested)
    if args.visualize and comparison_cam1 and comparison_cam2:
        visualize_comparison_results(comparison_cam1, comparison_cam2, output_vis_dir)

        # Create additional visualizations
        visualize_reasons_distribution(analyzed_csv_cam1, analyzed_csv_cam2, output_vis_dir)
        visualize_body_parts_distribution(analyzed_csv_cam1, analyzed_csv_cam2, output_vis_dir)

    logger.info("Image analysis steps completed successfully")

    return {
        'results_cam1': results_cam1,
        'results_cam2': results_cam2,
        'comparison_cam1': comparison_cam1,
        'comparison_cam2': comparison_cam2
    }


def visualize_aspect_ratio_distribution(data_cam1, data_cam2, output_dir):
    """
    Generate graphs to show the aspect ratio distribution
    """
    # Create a folder for visualizations
    vis_dir = output_dir / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    # Generate histogram graphs
    plt.figure(figsize=(12, 6))

    # Graph for camera 1
    plt.subplot(1, 2, 1)
    data_cam1['aspect_ratio'].hist(bins=30)
    plt.title('Aspect Ratio Distribution (Camera 1)')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    # Graph for camera 2
    plt.subplot(1, 2, 2)
    data_cam2['aspect_ratio'].hist(bins=30)
    plt.title('Aspect Ratio Distribution (Camera 2)')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(vis_dir / "aspect_ratio_distribution.png", dpi=300)
    plt.close()

    logger.info(f"Saved aspect ratio distribution graph to {vis_dir / 'aspect_ratio_distribution.png'}")


def visualize_comparison_results(comparison_cam1, comparison_cam2, output_dir):
    """
    Generate graphs to show comparison results
    """
    # Create comparison accuracy graph
    plt.figure(figsize=(10, 6))

    cameras = ['Camera 1', 'Camera 2']
    accuracy = [comparison_cam1.get('accuracy', 0), comparison_cam2.get('accuracy', 0)]
    precision = [comparison_cam1.get('precision', 0), comparison_cam2.get('precision', 0)]
    recall = [comparison_cam1.get('recall', 0), comparison_cam2.get('recall', 0)]
    f1_score = [comparison_cam1.get('f1_score', 0), comparison_cam2.get('f1_score', 0)]

    x = range(len(cameras))
    width = 0.2

    plt.bar([i - width * 1.5 for i in x], accuracy, width=width, label='Accuracy', color='blue')
    plt.bar([i - width * 0.5 for i in x], precision, width=width, label='Precision', color='green')
    plt.bar([i + width * 0.5 for i in x], recall, width=width, label='Recall', color='red')
    plt.bar([i + width * 1.5 for i in x], f1_score, width=width, label='F1-score', color='purple')

    plt.xlabel('Cameras')
    plt.ylabel('Score (%)')
    plt.title('Comparison of New Classification vs Original Annotations')
    plt.xticks(x, cameras)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_results.png", dpi=300)
    plt.close()

    logger.info(f"Saved comparison results graph to {output_dir / 'comparison_results.png'}")

    # Create Confusion Matrix graph
    plt.figure(figsize=(12, 5))

    # Confusion Matrix for camera 1
    plt.subplot(1, 2, 1)
    cm1 = np.array([
        [comparison_cam1.get('true_positive', 0), comparison_cam1.get('false_negative', 0)],
        [comparison_cam1.get('false_positive', 0), comparison_cam1.get('true_negative', 0)]
    ])

    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Good', 'Predicted Junk'],
                yticklabels=['Actual Good', 'Actual Junk'])
    plt.title('Confusion Matrix - Camera 1')

    # Confusion Matrix for camera 2
    plt.subplot(1, 2, 2)
    cm2 = np.array([
        [comparison_cam2.get('true_positive', 0), comparison_cam2.get('false_negative', 0)],
        [comparison_cam2.get('false_positive', 0), comparison_cam2.get('true_negative', 0)]
    ])

    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Good', 'Predicted Junk'],
                yticklabels=['Actual Good', 'Actual Junk'])
    plt.title('Confusion Matrix - Camera 2')

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    logger.info(f"Saved Confusion Matrix graph to {output_dir / 'confusion_matrix.png'}")


def visualize_reasons_distribution(csv_cam1, csv_cam2, output_dir):
    """
    สร้างกราฟแสดงการกระจายตัวของเหตุผลที่รูปภาพเป็น junk
    """
    try:
        # อ่านไฟล์ CSV
        df_cam1 = pd.read_csv(csv_cam1)
        df_cam2 = pd.read_csv(csv_cam2)

        # กรองเฉพาะรูปที่เป็น junk
        junk_cam1 = df_cam1[df_cam1['class'] == 'junk']
        junk_cam2 = df_cam2[df_cam2['class'] == 'junk']

        # แบ่งเหตุผลเป็นหมวดหมู่ (เนื่องจากเหตุผลอาจมีหลายข้อในแต่ละรูป)
        reason_categories = [
            "Aspect ratio", "No head", "less than 70% visible",
            "No torso", "less than 80% visible", "No legs", "less than 40% visible",
            "Light sources", "Body occlusion"
        ]

        # นับจำนวนรูปภาพตามเหตุผล
        reason_counts_cam1 = []
        reason_counts_cam2 = []

        for category in reason_categories:
            count1 = junk_cam1['reason'].str.contains(category, case=False).sum() if 'reason' in junk_cam1.columns else 0
            count2 = junk_cam2['reason'].str.contains(category, case=False).sum() if 'reason' in junk_cam2.columns else 0
            reason_counts_cam1.append(count1)
            reason_counts_cam2.append(count2)

        # สร้างกราฟ
        plt.figure(figsize=(14, 8))
        x = np.arange(len(reason_categories))
        width = 0.35

        plt.bar(x - width/2, reason_counts_cam1, width, label='Camera 1')
        plt.bar(x + width/2, reason_counts_cam2, width, label='Camera 2')

        plt.xlabel('Reasons')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Reasons for Junk Images')
        plt.xticks(x, reason_categories, rotation=45, ha='right')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "reasons_distribution.png", dpi=300)
        plt.close()

        logger.info(f"Saved reasons distribution graph to {output_dir / 'reasons_distribution.png'}")

    except Exception as e:
        logger.error(f"Error creating reasons distribution graph: {str(e)}")


def visualize_body_parts_distribution(csv_cam1, csv_cam2, output_dir):
    """
    สร้างกราฟแสดงการกระจายตัวของเปอร์เซ็นต์ส่วนต่างๆ ของร่างกาย
    """
    try:
        # อ่านไฟล์ CSV
        df_cam1 = pd.read_csv(csv_cam1)
        df_cam2 = pd.read_csv(csv_cam2)

        # Check if the necessary columns exist - now adapted for YOLO analysis
        body_part_columns = ['head_percentage', 'torso_percentage', 'legs_percentage']
        if not all(col in df_cam1.columns for col in body_part_columns) or not all(col in df_cam2.columns for col in body_part_columns):
            logger.warning("Body part percentage columns not found in the CSV files")
            return

        # สร้างกราฟสำหรับ head_percentage
        plt.figure(figsize=(10, 6))

        sns.histplot(df_cam1['head_percentage'].dropna(), kde=True, label='Camera 1', alpha=0.6)
        sns.histplot(df_cam2['head_percentage'].dropna(), kde=True, label='Camera 2', alpha=0.6)

        plt.xlabel('Head Percentage')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Head Percentage')
        plt.axvline(x=70, color='red', linestyle='--', label='Minimum Threshold (70%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "head_percentage_distribution.png", dpi=300)
        plt.close()

        # สร้างกราฟสำหรับ torso_percentage
        plt.figure(figsize=(10, 6))

        sns.histplot(df_cam1['torso_percentage'].dropna(), kde=True, label='Camera 1', alpha=0.6)
        sns.histplot(df_cam2['torso_percentage'].dropna(), kde=True, label='Camera 2', alpha=0.6)

        plt.xlabel('Torso Percentage')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Torso Percentage')
        plt.axvline(x=80, color='red', linestyle='--', label='Minimum Threshold (80%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "torso_percentage_distribution.png", dpi=300)
        plt.close()

        # สร้างกราฟสำหรับ legs_percentage
        plt.figure(figsize=(10, 6))

        sns.histplot(df_cam1['legs_percentage'].dropna(), kde=True, label='Camera 1', alpha=0.6)
        sns.histplot(df_cam2['legs_percentage'].dropna(), kde=True, label='Camera 2', alpha=0.6)

        plt.xlabel('Legs Percentage')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Legs Percentage')
        plt.axvline(x=40, color='red', linestyle='--', label='Minimum Threshold (40%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "legs_percentage_distribution.png", dpi=300)
        plt.close()

        logger.info(f"Saved body parts percentage distribution graphs to {output_dir}")

    except Exception as e:
        logger.error(f"Error creating body parts percentage distribution graphs: {str(e)}")


def main():
    """
    The main function for program execution
    """
    # Parse arguments
    args = parse_arguments()

    # Display settings
    logger.info("Starting person image analysis program")
    logger.info(f"Camera 1 CSV: {args.csv1}")
    logger.info(f"Camera 2 CSV: {args.csv2}")
    logger.info(f"Camera 1 Images: {args.images1}")
    logger.info(f"Camera 2 Images: {args.images2}")
    logger.info(f"Output folder: {args.output}")
    logger.info(f"Minimum aspect ratio: {args.min_aspect_ratio}")
    logger.info(f"YOLO model: {args.yolo_model}")

    # Create output folder
    os.makedirs(args.output, exist_ok=True)

    # Start timer
    start_time = time.time()

    # Perform steps
    if not args.skip_preparation:
        prepared_csv_cam1, prepared_csv_cam2, prepared_csv_combined = prepare_data(args)
    else:
        # If skipping data preparation, use default paths for already prepared files
        output_csv_dir = Path(args.output) / "csv"
        prepared_csv_cam1 = output_csv_dir / "prepared_cam1.csv"
        prepared_csv_cam2 = output_csv_dir / "prepared_cam2.csv"

        if not prepared_csv_cam1.exists() or not prepared_csv_cam2.exists():
            logger.warning("No prepared files found, starting data preparation...")
            prepared_csv_cam1, prepared_csv_cam2, prepared_csv_combined = prepare_data(args)

    # If only performing data preparation
    if args.only_preparation:
        logger.info("Only data preparation step completed. Program terminated.")
    else:
        # Analyze images
        results = analyze_images(args, prepared_csv_cam1, prepared_csv_cam2)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Process completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise