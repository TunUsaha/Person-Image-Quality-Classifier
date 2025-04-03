import pandas as pd
import os
import cv2
import numpy as np
from pathlib import Path
import logging

# Configure logging to track operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparation:
    def __init__(self, csv_cam1_path, csv_cam2_path, images_folder_cam1, images_folder_cam2):
        """
        Prepare data for analysis.

        Args:
            csv_cam1_path (str): Path to the CSV file for camera 1
            csv_cam2_path (str): Path to the CSV file for camera 2
            images_folder_cam1 (str): Folder containing images from camera 1
            images_folder_cam2 (str): Folder containing images from camera 2
        """
        self.csv_cam1_path = csv_cam1_path
        self.csv_cam2_path = csv_cam2_path
        self.images_folder_cam1 = images_folder_cam1
        self.images_folder_cam2 = images_folder_cam2

        # Define paths as Path objects for easier usage
        self.images_path_cam1 = Path(images_folder_cam1)
        self.images_path_cam2 = Path(images_folder_cam2)

        # DataFrames to store data
        self.data_cam1 = None
        self.data_cam2 = None

        # Paths for missing images
        self.missing_images = []

    def load_csv_data(self):
        """
        Load data from CSV files for both cameras.
        """
        logger.info("Loading data from CSV files...")

        try:
            # Try reading the file using comma delimiter first
            self.data_cam1 = pd.read_csv(self.csv_cam1_path)
            logger.info(f"Successfully loaded data from camera 1: {self.csv_cam1_path}")
        except Exception as e:
            # If unsuccessful, try using tab delimiter
            try:
                self.data_cam1 = pd.read_csv(self.csv_cam1_path, delimiter='\t')
                logger.info(f"Successfully loaded data from camera 1 with tab delimiter: {self.csv_cam1_path}")
            except Exception as e2:
                logger.error(f"Unable to load data from camera 1: {str(e2)}")
                raise

        try:
            # Try reading the file using comma delimiter first
            self.data_cam2 = pd.read_csv(self.csv_cam2_path)
            logger.info(f"Successfully loaded data from camera 2: {self.csv_cam2_path}")
        except Exception as e:
            # If unsuccessful, try using tab delimiter
            try:
                self.data_cam2 = pd.read_csv(self.csv_cam2_path, delimiter='\t')
                logger.info(f"Successfully loaded data from camera 2 with tab delimiter: {self.csv_cam2_path}")
            except Exception as e2:
                logger.error(f"Unable to load data from camera 2: {str(e2)}")
                raise

        # Display sample data
        logger.info(f"Sample data from camera 1:\n{self.data_cam1.head()}")
        logger.info(f"Sample data from camera 2:\n{self.data_cam2.head()}")

        # Add a column to indicate the camera (useful for merging data)
        self.data_cam1['camera'] = 'cam1'
        self.data_cam2['camera'] = 'cam2'

        return self.data_cam1, self.data_cam2

    def check_image_files(self):
        """
        Check if the images in the cropped_filename column actually exist.
        """
        logger.info("Checking image files...")

        # Prepare lists to store filenames of missing images
        missing_cam1 = []
        missing_cam2 = []

        # Check images from camera 1
        for filename in self.data_cam1['cropped_filename']:
            file_path = self.images_path_cam1 / filename
            if not file_path.exists():
                missing_cam1.append(filename)

        # Check images from camera 2
        for filename in self.data_cam2['cropped_filename']:
            file_path = self.images_path_cam2 / filename
            if not file_path.exists():
                missing_cam2.append(filename)

        # Save and display results
        self.missing_images = {'cam1': missing_cam1, 'cam2': missing_cam2}

        logger.info(f"Number of missing images in camera 1: {len(missing_cam1)}")
        logger.info(f"Number of missing images in camera 2: {len(missing_cam2)}")

        if missing_cam1:
            logger.warning(f"Examples of missing images in camera 1: {missing_cam1[:5]}")
        if missing_cam2:
            logger.warning(f"Examples of missing images in camera 2: {missing_cam2[:5]}")

        return self.missing_images

    def add_aspect_ratio_column(self):
        """
        Add an aspect_ratio column by calculating it from the actual images.
        """
        logger.info("Calculating aspect ratio for images...")

        # Add new columns
        self.data_cam1['aspect_ratio'] = None
        self.data_cam1['width'] = None
        self.data_cam1['height'] = None

        self.data_cam2['aspect_ratio'] = None
        self.data_cam2['width'] = None
        self.data_cam2['height'] = None

        # Calculate for camera 1
        for index, row in self.data_cam1.iterrows():
            filename = row['cropped_filename']
            file_path = self.images_path_cam1 / filename

            if file_path.exists():
                try:
                    img = cv2.imread(str(file_path))
                    height, width = img.shape[:2]
                    aspect_ratio = height / width

                    self.data_cam1.at[index, 'aspect_ratio'] = aspect_ratio
                    self.data_cam1.at[index, 'width'] = width
                    self.data_cam1.at[index, 'height'] = height
                except Exception as e:
                    logger.error(f"Error reading image {filename}: {str(e)}")

        # Calculate for camera 2
        for index, row in self.data_cam2.iterrows():
            filename = row['cropped_filename']
            file_path = self.images_path_cam2 / filename

            if file_path.exists():
                try:
                    img = cv2.imread(str(file_path))
                    height, width = img.shape[:2]
                    aspect_ratio = height / width

                    self.data_cam2.at[index, 'aspect_ratio'] = aspect_ratio
                    self.data_cam2.at[index, 'width'] = width
                    self.data_cam2.at[index, 'height'] = height
                except Exception as e:
                    logger.error(f"Error reading image {filename}: {str(e)}")

        # Display aspect ratio statistics
        logger.info(f"Aspect ratio statistics for camera 1:\n{self.data_cam1['aspect_ratio'].describe()}")
        logger.info(f"Aspect ratio statistics for camera 2:\n{self.data_cam2['aspect_ratio'].describe()}")

        return self.data_cam1, self.data_cam2

    def combine_data(self):
        """
        Combine data from both cameras.
        """
        logger.info("Combining data from both cameras...")
        combined_data = pd.concat([self.data_cam1, self.data_cam2], ignore_index=True)
        logger.info(f"Successfully combined data. Total rows: {len(combined_data)}")

        return combined_data

    def save_prepared_data(self, output_path_cam1, output_path_cam2, output_path_combined):
        """
        Save the prepared data.

        Args:
            output_path_cam1 (str): Path to save data for camera 1
            output_path_cam2 (str): Path to save data for camera 2
            output_path_combined (str): Path to save combined data
        """
        logger.info("Saving prepared data...")

        # Create folders if they don't exist
        os.makedirs(os.path.dirname(output_path_cam1), exist_ok=True)
        os.makedirs(os.path.dirname(output_path_cam2), exist_ok=True)
        os.makedirs(os.path.dirname(output_path_combined), exist_ok=True)

        # Save the data
        self.data_cam1.to_csv(output_path_cam1, index=False)
        self.data_cam2.to_csv(output_path_cam2, index=False)

        # Combine and save the merged data
        combined_data = self.combine_data()
        combined_data.to_csv(output_path_combined, index=False)

        logger.info(f"Saved camera 1 data to: {output_path_cam1}")
        logger.info(f"Saved camera 2 data to: {output_path_cam2}")
        logger.info(f"Saved combined data to: {output_path_combined}")

        return output_path_cam1, output_path_cam2, output_path_combined

    def analyze_aspect_ratio_distribution(self):
        """
        Analyze the distribution of aspect ratios.
        """
        logger.info("Analyzing the distribution of aspect ratios...")

        # Check aspect ratio distribution
        aspect_ratios_cam1 = self.data_cam1['aspect_ratio'].dropna()
        aspect_ratios_cam2 = self.data_cam2['aspect_ratio'].dropna()

        # Calculate basic statistics
        stats_cam1 = {
            'min': aspect_ratios_cam1.min(),
            'max': aspect_ratios_cam1.max(),
            'mean': aspect_ratios_cam1.mean(),
            'median': aspect_ratios_cam1.median(),
            'std': aspect_ratios_cam1.std(),
            'percentiles': {
                '5%': aspect_ratios_cam1.quantile(0.05),
                '25%': aspect_ratios_cam1.quantile(0.25),
                '75%': aspect_ratios_cam1.quantile(0.75),
                '95%': aspect_ratios_cam1.quantile(0.95)
            }
        }

        stats_cam2 = {
            'min': aspect_ratios_cam2.min(),
            'max': aspect_ratios_cam2.max(),
            'mean': aspect_ratios_cam2.mean(),
            'median': aspect_ratios_cam2.median(),
            'std': aspect_ratios_cam2.std(),
            'percentiles': {
                '5%': aspect_ratios_cam2.quantile(0.05),
                '25%': aspect_ratios_cam2.quantile(0.25),
                '75%': aspect_ratios_cam2.quantile(0.75),
                '95%': aspect_ratios_cam2.quantile(0.95)
            }
        }

        logger.info(f"Aspect ratio statistics for camera 1: {stats_cam1}")
        logger.info(f"Aspect ratio statistics for camera 2: {stats_cam2}")

        return stats_cam1, stats_cam2


# Example usage
def prepare_data_example():
    # Define paths
    csv_cam1_path = "image_annotations_cropped_cam1.csv"
    csv_cam2_path = "image_annotations_cropped_cam2.csv"
    images_folder_cam1 = "/Users/ahasunut/PycharmProjects/PythonProject/YOLOv11_con70_person_sequence_cam1"
    images_folder_cam2 = "/Users/ahasunut/PycharmProjects/PythonProject/YOLOv11_con70_person_sequence_cam2"

    # Create an instance of the DataPreparation class
    data_prep = DataPreparation(csv_cam1_path, csv_cam2_path, images_folder_cam1, images_folder_cam2)

    # Load CSV data
    data_cam1, data_cam2 = data_prep.load_csv_data()

    # Check image files
    missing_images = data_prep.check_image_files()

    # Add aspect ratio
    data_cam1, data_cam2 = data_prep.add_aspect_ratio_column()

    # Analyze aspect ratio distribution
    stats_cam1, stats_cam2 = data_prep.analyze_aspect_ratio_distribution()

    # Save prepared data
    output_path_cam1 = "output/prepared_data_cam1.csv"
    output_path_cam2 = "output/prepared_data_cam2.csv"
    output_path_combined = "output/prepared_data_combined.csv"

    data_prep.save_prepared_data(output_path_cam1, output_path_cam2, output_path_combined)

    print("Data preparation completed successfully")


if __name__ == "__main__":
    prepare_data_example()