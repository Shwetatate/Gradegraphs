import pandas as pd
import numpy as np
from .utils import extract_max_marks_from_header
from typing import List, Dict
new rep

# Minimal inline implementations to remove dependency on removed processor.py
def get_dynamic_subject_recommendations(subject_name, student_marks, max_marks, weak_count=1):
    performance_percentage = (student_marks / max_marks * 100) if max_marks > 0 else 0
    subject_lower = str(subject_name).lower()
    recs: List[Dict] = []
    def urgency(perf):
        if perf < 40:
            return 'Critical', 'intensive', 'daily'
        if perf < 60:
            return 'High', 'focused', '4-5 times per week'
        if perf < 75:
            return 'Medium', 'targeted', '3 times per week'
        return 'Low', 'enhancement', 'twice per week'
    urg, intensity, freq = urgency(performance_percentage)
    recs.append({'title': f'{urg} Study Plan', 'description': f'Improve from {performance_percentage:.1f}%', 'action': f'Allocate sessions {freq}'})
    return recs


def get_subject_recommendations(subject_name, weak_count=1):
    return [
        {'title': 'Personalized Learning Path', 'description': f'Customized path for {subject_name}', 'action': 'Create tailored study plan'},
        {'title': 'Progress Tracking', 'description': 'Weekly checks', 'action': 'Track with mini-tests'},
    ]


def generate_individual_student_recommendations(student_data, df, subjects):
    name = student_data.get('Name', 'Student') if isinstance(student_data, dict) else 'Student'
    return [
        {'title': f'Performance Plan for {name}', 'description': 'Target weaker areas', 'action': 'Structured weekly schedule'}
    ]


def create_class_improvement_plan(df, subjects):
    return {'overview': {'total_students': len(df)}, 'priority_areas': [], 'strategies': [], 'timeline': {}}


def create_comprehensive_report(df, subjects, subject_difficulty):
    return df


def get_threshold_based_recommendations(df, subject_name, threshold_value, exam_type=None):
    try:
        def _extract_max_marks_from_col(col_name):
            try:
                return extract_max_marks_from_header(col_name)
            except Exception:
                return None

        subject_cols = []
        for col in df.columns:
            col_upper = str(col).upper()
            subject_upper = subject_name.upper()
            if subject_upper in col_upper:
                if exam_type:
                    exam_upper = exam_type.upper()
                    if exam_upper in col_upper:
                        subject_cols.append(col)
                else:
                    subject_cols.append(col)

        if not subject_cols:
            return {
                'error': True,
                'message': f'No columns found for subject: {subject_name}',
                'students': []
            }

        col_to_max_marks = {}
        for col in subject_cols:
            detected = _extract_max_marks_from_col(col)
            if detected is None:
                # Conservative default when header lacks numbers
                detected = 100
            col_to_max_marks[col] = detected

        detected_max_marks = max(col_to_max_marks.values()) if col_to_max_marks else 100

        if threshold_value > detected_max_marks:
            return {
                'error': True,
                'message': f'Threshold value ({threshold_value}) exceeds detected maximum marks ({detected_max_marks}) for {subject_name} {exam_type if exam_type else ""}',
                'max_marks': detected_max_marks,
                'students': []
            }

        students_above_threshold = []

        possible_name_cols = ['Name', 'Student Name', 'Student_Name', 'STUDENT NAME']
        possible_roll_cols = ['Roll No', 'Roll_No', 'ROLL NO', 'SR.No', 'SR NO', 'Sr No']

        detected_name_col = next((c for c in possible_name_cols if c in df.columns), None)
        detected_roll_col = next((c for c in possible_roll_cols if c in df.columns), None)

        for idx, row in df.iterrows():
            student_name = row.get(detected_name_col) if detected_name_col else None
            if pd.isna(student_name) or student_name is None or str(student_name).strip() == '':
                student_name = f"Student {idx+1}"

            roll_no = row.get(detected_roll_col) if detected_roll_col else None
            if pd.isna(roll_no) or roll_no is None or str(roll_no).strip() == '':
                roll_no = f"{idx+1}"

            for col in subject_cols:
                marks = row.get(col, 0)
                try:
                    marks = float(marks) if pd.notna(marks) else 0
                except (ValueError, TypeError):
                    marks = 0

                if marks >= threshold_value:
                    col_upper = str(col).upper()
                    detected_exam_type = 'Unknown'
                    if 'ESE' in col_upper:
                        detected_exam_type = 'ESE'
                    elif 'ISE' in col_upper:
                        detected_exam_type = 'ISE'
                    elif 'MSE' in col_upper:
                        detected_exam_type = 'MSE'
                    elif 'PRACTICAL' in col_upper or 'PR' in col_upper:
                        detected_exam_type = 'PRACTICAL'
                    elif 'TW' in col_upper:
                        detected_exam_type = 'TW'

                    col_max = col_to_max_marks.get(col, detected_max_marks)
                    performance_pct = (marks / col_max * 100) if col_max else 0

                    students_above_threshold.append({
                        'Student Name': student_name,
                        'Roll No': roll_no,
                        'Subject': subject_name,
                        'Exam Type': detected_exam_type,
                        'Column': col,
                        'Marks': marks,
                        'Threshold': threshold_value,
                        'Max Marks': col_max,
                        'Performance': f"{marks}/{col_max} ({performance_pct:.1f}%)"
                    })

        students_above_threshold.sort(key=lambda x: x['Marks'], reverse=True)

        return {
            'error': False,
            'message': f'Found {len(students_above_threshold)} students above threshold {threshold_value}',
            'max_marks': detected_max_marks,
            'threshold': threshold_value,
            'exam_type': exam_type,
            'students': students_above_threshold,
            'total_students': len(df),
            'percentage_above_threshold': (len(students_above_threshold) / len(df)) * 100 if len(df) > 0 else 0
        }
    except Exception as e:
        return {
            'error': True,
            'message': f'Error processing threshold recommendations: {str(e)}',
            'students': []
        }


