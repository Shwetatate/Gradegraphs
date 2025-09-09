import re
import pandas as pd
import logging
import os
from pathlib import Path

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicMarksExtractor:
    def __init__(self, config_file_path=None):
        """
        Initialize the extractor with optional configuration file path

        Args:
            config_file_path (str): Path to Excel file containing assessment type configurations
        """
        self.config_file_path = config_file_path
        self.assessment_defaults = {}
        self.practical_keywords = []
        self.special_practical_keywords = {}

        # Load configuration from Excel file
        if config_file_path and os.path.exists(config_file_path):
            self.load_configuration_from_excel()
        else:
            # Create default configuration if no file provided
            self.create_default_configuration()

    def load_configuration_from_excel(self):
        """
        Load assessment type configurations from Excel file
        Expected Excel structure:
        Sheet 1 'AssessmentTypes': assessment_type | default_marks | description
        Sheet 2 'PracticalKeywords': keyword | marks_multiplier
        Sheet 3 'SpecialKeywords': keyword | default_marks
        """
        try:
            logger.info(f"Loading configuration from: {self.config_file_path}")

            # Load assessment types configuration
            assessment_df = pd.read_excel(self.config_file_path, sheet_name='AssessmentTypes')
            self.assessment_defaults = dict(zip(
                assessment_df['assessment_type'].str.upper(),
                assessment_df['default_marks']
            ))
            logger.info(f"Loaded {len(self.assessment_defaults)} assessment types")

            # Load practical keywords
            try:
                practical_df = pd.read_excel(self.config_file_path, sheet_name='PracticalKeywords')
                self.practical_keywords = practical_df['keyword'].str.upper().tolist()
                logger.info(f"Loaded {len(self.practical_keywords)} practical keywords")
            except Exception as e:
                logger.warning(f"Could not load PracticalKeywords sheet: {e}")
                self.practical_keywords = ['PRACTICAL', 'PR', 'LAB', 'EXPERIMENT', 'PROJECT', 'VIVA']

            # Load special practical keywords with custom marks
            try:
                special_df = pd.read_excel(self.config_file_path, sheet_name='SpecialKeywords')
                self.special_practical_keywords = dict(zip(
                    special_df['keyword'].str.upper(),
                    special_df['default_marks']
                ))
                logger.info(f"Loaded {len(self.special_practical_keywords)} special keywords")
            except Exception as e:
                logger.warning(f"Could not load SpecialKeywords sheet: {e}")
                self.special_practical_keywords = {'DESIGN': 50, 'INNOVATION': 50, 'MAJOR': 50}

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
            self.create_default_configuration()

    def create_default_configuration(self):
        """
        Create default configuration when no Excel file is provided
        """
        self.assessment_defaults = {
            'ESE': 80,
            'ISE': 20,
            'MSE': 20,
            'TW': 25,
            'PRACTICAL': 100,
            'PR': 100,
            'LAB': 25,
            'VIVA': 10,
            'PROJECT': 50,
            'ASSIGNMENT': 10,
            'QUIZ': 10,
            'ATTENDANCE': 5
        }

        self.practical_keywords = ['PRACTICAL', 'PR', 'LAB', 'EXPERIMENT', 'PROJECT', 'VIVA']
        self.special_practical_keywords = {'DESIGN': 50, 'INNOVATION': 50, 'MAJOR': 50}

        logger.info("Using default configuration")

    def save_configuration_template(self, output_path="marks_configuration_template.xlsx"):
        """
        Save a template Excel file for configuration
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Assessment Types sheet
            assessment_data = {
                'assessment_type': list(self.assessment_defaults.keys()),
                'default_marks': list(self.assessment_defaults.values()),
                'description': [
                    'End Semester Exam', 'Internal Semester Exam', 'Mid Semester Exam',
                    'Term Work', 'Practical', 'Practical', 'Laboratory', 'Viva Voce',
                    'Project', 'Assignment', 'Quiz', 'Attendance'
                ]
            }
            pd.DataFrame(assessment_data).to_excel(writer, sheet_name='AssessmentTypes', index=False)

            # Practical Keywords sheet
            practical_data = {
                'keyword': self.practical_keywords,
                'marks_multiplier': [1.0] * len(self.practical_keywords)
            }
            pd.DataFrame(practical_data).to_excel(writer, sheet_name='PracticalKeywords', index=False)

            # Special Keywords sheet
            special_data = {
                'keyword': list(self.special_practical_keywords.keys()),
                'default_marks': list(self.special_practical_keywords.values())
            }
            pd.DataFrame(special_data).to_excel(writer, sheet_name='SpecialKeywords', index=False)

        logger.info(f"Configuration template saved to: {output_path}")

    def is_practical_column(self, col_name):
        """
        Enhanced practical column detection using dynamic keywords
        """
        if not col_name or pd.isna(col_name):
            return False

        col_upper = str(col_name).upper()

        # Check against dynamically loaded practical keywords
        for keyword in self.practical_keywords:
            if keyword in col_upper:
                return True

        # Additional pattern-based checking
        practical_patterns = [
            r'\s+PR\s*$',
            r'\s+PR\s*\(',
            r'PR\)\s*$',
            r'\bPR\s*\(\d+\)',
        ]

        for pattern in practical_patterns:
            if re.search(pattern, col_upper):
                return True

        # Exclude TW columns that don't have PR
        if 'TW' in col_upper and 'PR' not in col_upper:
            return False

        return False

    def get_default_marks_by_type(self, col_upper):
        """
        Get default marks based on assessment type keywords from Excel configuration
        """
        # Check for assessment type in column name using dynamic configuration
        for assessment_type, default_mark in self.assessment_defaults.items():
            if assessment_type in col_upper:
                logger.info(f"Found assessment type '{assessment_type}' with {default_mark} marks")
                return default_mark

        # Special handling for practical columns using dynamic keywords
        if self.is_practical_column(col_upper):
            # Check special practical keywords first
            for special_keyword, marks in self.special_practical_keywords.items():
                if special_keyword in col_upper:
                    logger.info(f"Found special practical keyword '{special_keyword}' with {marks} marks")
                    return marks

            # Default practical marks from configuration (non-static; high placeholder until header parsed)
            practical_default = self.assessment_defaults.get('PRACTICAL', 100)
            logger.info(f"Using default practical marks: {practical_default}")
            return practical_default

        # Default fallback
        logger.info("Using default fallback marks: 100")
        return 100

    def extract_marks_from_header(self, col_name):
        """
        Enhanced marks extraction with better parsing and fallback logic
        """
        if not col_name or pd.isna(col_name):
            return 100

        col_str = str(col_name).strip()
        col_upper = col_str.upper()

        logger.info(f"Processing column: {col_str}")

        # Priority 1: Extract from parentheses (most reliable)
        parentheses_match = re.search(r'\((\d+)\)', col_str)
        if parentheses_match:
            marks = int(parentheses_match.group(1))
            logger.info(f"Extracted from parentheses: {marks}")
            return marks

        # Priority 2: Extract from square brackets
        bracket_match = re.search(r'\[(\d+)\]', col_str)
        if bracket_match:
            marks = int(bracket_match.group(1))
            logger.info(f"Extracted from brackets: {marks}")
            return marks

        # Priority 3: Look for marks after keywords (using dynamic assessment types)
        for assessment_type in self.assessment_defaults.keys():
            pattern = rf'{assessment_type}[\s\-_]*(\d+)'
            match = re.search(pattern, col_upper)
            if match:
                marks = int(match.group(1))
                logger.info(f"Extracted from {assessment_type} pattern: {marks}")
                return marks

        # Priority 4: Find all numbers and use contextual logic
        all_numbers = re.findall(r'\d+', col_str)
        if all_numbers:
            numbers = [int(num) for num in all_numbers]

            # If multiple numbers, try to identify which one represents marks
            if len(numbers) > 1:
                # Look for marks that match our configuration
                config_marks = list(self.assessment_defaults.values()) + list(self.special_practical_keywords.values())
                for mark in config_marks:
                    if mark in numbers:
                        logger.info(f"Selected configured mark value: {mark}")
                        return mark

                # Use the largest reasonable number (likely the total marks)
                reasonable_marks = [num for num in numbers if 5 <= num <= 200]
                if reasonable_marks:
                    marks = max(reasonable_marks)
                    logger.info(f"Selected max reasonable number: {marks}")
                    return marks

            # Single number or fallback to last number
            marks = numbers[-1]
            logger.info(f"Using last number found: {marks}")
            return marks

        # Priority 5: Default values based on assessment type (from Excel configuration)
        default_marks = self.get_default_marks_by_type(col_upper)
        logger.info(f"Using default marks for type: {default_marks}")
        return default_marks

    def identify_assessment_type(self, col_name):
        """
        Identify the type of assessment from column name using dynamic configuration
        """
        if not col_name or pd.isna(col_name):
            return 'UNKNOWN'

        col_upper = str(col_name).upper()

        # Check against dynamic assessment types
        for assessment_type in self.assessment_defaults.keys():
            if assessment_type in col_upper:
                return assessment_type

        return 'General Assessment'

    def process_dataframe_headers(self, df):
        """
        Process all column headers in a DataFrame and extract marks
        """
        header_info = {}

        for col in df.columns:
            marks = self.extract_marks_from_header(col)
            is_practical = self.is_practical_column(col)

            header_info[col] = {
                'original_name': col,
                'max_marks': marks,
                'is_practical': is_practical,
                'assessment_type': self.identify_assessment_type(col)
            }

        return header_info

    def update_configuration_from_dataframe(self, df, output_path="updated_marks_config.xlsx"):
        """
        Analyze a DataFrame and suggest/update configuration based on found patterns
        """
        found_patterns = {}

        for col in df.columns:
            col_upper = str(col).upper()

            # Extract numbers from headers
            numbers = re.findall(r'\d+', str(col))
            if numbers:
                # Try to identify assessment type
                for assessment_type in self.assessment_defaults.keys():
                    if assessment_type in col_upper:
                        found_patterns[assessment_type] = int(numbers[-1])
                        break

        # Update configuration with found patterns
        for pattern, marks in found_patterns.items():
            if pattern in self.assessment_defaults:
                old_marks = self.assessment_defaults[pattern]
                self.assessment_defaults[pattern] = marks
                logger.info(f"Updated {pattern}: {old_marks} -> {marks}")

        # Save updated configuration
        self.save_configuration_template(output_path)
        logger.info(f"Updated configuration saved to: {output_path}")

# Convenience functions for backward compatibility
def create_marks_extractor(config_file_path=None):
    """
    Factory function to create a DynamicMarksExtractor instance
    """
    return DynamicMarksExtractor(config_file_path)

# Module-level singleton extractor and wrappers for legacy imports
_default_extractor = DynamicMarksExtractor()

def extract_max_marks_from_header(col_name):
    return _default_extractor.extract_marks_from_header(col_name)

def is_practical_column(col_name):
    return _default_extractor.is_practical_column(col_name)

def demo_with_excel_config():
    """
    Demonstrate the dynamic configuration system
    """
    # Create an instance and save a template
    extractor = DynamicMarksExtractor()
    extractor.save_configuration_template("marks_config_template.xlsx")

    print("=== Demo: Dynamic Marks Extraction ===")
    print("1. Template Excel file created: marks_config_template.xlsx")
    print("2. Edit the Excel file to customize your assessment types and marks")
    print("3. Use the extractor like this:")
    print()

    # Sample usage
    sample_headers = [
        "Mathematics ISE (20)",
        "Physics ESE [80]",
        "Chemistry Practical PR (25)",
        "English TW 50",
        "Computer Science Project",
        "Biology Lab Viva (10)"
    ]

    for header in sample_headers:
        marks = extractor.extract_marks_from_header(header)
        assessment_type = extractor.identify_assessment_type(header)
        print(f"'{header}' -> {marks} marks ({assessment_type})")

# Example usage
if __name__ == "__main__":
    # Demo the system
    demo_with_excel_config()

    # Example of using with custom Excel configuration
    # extractor = DynamicMarksExtractor("your_custom_config.xlsx")
    # df = pd.read_csv("your_student_data.csv")
    # header_info = extractor.process_dataframe_headers(df)