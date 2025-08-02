# FILE 3: src/utils/helpers.py
# ============================================
"""Minimal helper utilities module"""

import pandas as pd
import numpy as np
import streamlit as st

class DataProcessor:
    """Data processing utilities"""
    
    @staticmethod
    def validate_data_integrity(datasets):
        """Validate data integrity"""
        issues = {}
        for name, df in datasets.items():
            if df.empty:
                issues[name] = ['Empty dataset']
        return issues
    
    @staticmethod
    def create_summary_statistics(data, numeric_cols):
        """Create summary statistics"""
        return data[numeric_cols].describe()

class StreamlitHelpers:
    """Streamlit-specific helpers"""
    
    @staticmethod
    def create_metric_card(label, value, delta=None, format_str="{:.2f}"):
        """Create metric card"""
        if delta:
            st.metric(label, format_str.format(value), delta=format_str.format(delta))
        else:
            st.metric(label, format_str.format(value))
    
    @staticmethod
    def create_download_button(data, filename, button_text="Download CSV"):
        """Create download button"""
        csv = data.to_csv(index=False)
        st.download_button(
            label=button_text,
            data=csv,
            file_name=filename,
            mime='text/csv'
        )

class AnalyticsHelpers:
    """Analytics helper functions"""
    
    @staticmethod
    def calculate_engagement_index(data):
        """Calculate engagement index"""
        # Simple average of available metrics
        metrics = ['quest_completion_rate', 'skill_progression']
        available = [m for m in metrics if m in data.columns]
        if available:
            return data[available].mean(axis=1)
        return pd.Series([0.5] * len(data))
    
    @staticmethod
    def identify_at_risk_students(data, thresholds=None):
        """Identify at-risk students"""
        if thresholds is None:
            thresholds = {'engagement_score': 0.3, 'days_active': 5}
        
        at_risk = pd.DataFrame()
        at_risk['student_id'] = data['student_id']
        at_risk['at_risk'] = False
        
        for metric, threshold in thresholds.items():
            if metric in data.columns:
                at_risk['at_risk'] |= (data[metric] < threshold)
        
        return at_risk[at_risk['at_risk']]
