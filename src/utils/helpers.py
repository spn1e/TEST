## src/utils/helpers.py
```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st
from datetime import datetime, timedelta
import json
import yaml

class DataProcessor:
    """Utility functions for data processing and validation"""
    
    @staticmethod
    def validate_data_integrity(datasets: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Validate data integrity across all datasets"""
        issues = {dataset: [] for dataset in datasets.keys()}
        
        # Check for required columns
        required_columns = {
            'students': ['student_id', 'grade_level', 'learning_style'],
            'movements': ['student_id', 'timestamp', 'x', 'y', 'z', 'zone'],
            'buildings': ['student_id', 'timestamp', 'action', 'block_type'],
            'quests': ['student_id', 'quest_name', 'completed', 'score'],
            'collaborations': ['student_1', 'student_2', 'interaction_type'],
            'learning_analytics': ['student_id', 'engagement_score', 'skill_progression']
        }
        
        for dataset_name, df in datasets.items():
            if dataset_name in required_columns:
                missing_cols = set(required_columns[dataset_name]) - set(df.columns)
                if missing_cols:
                    issues[dataset_name].append(f"Missing columns: {missing_cols}")
        
        # Check for data types
        for dataset_name, df in datasets.items():
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                null_cols = null_counts[null_counts > 0].index.tolist()
                issues[dataset_name].append(f"Null values in: {null_cols}")
            
            # Check for duplicate student IDs in students table
            if dataset_name == 'students':
                duplicates = df['student_id'].duplicated().sum()
                if duplicates > 0:
                    issues[dataset_name].append(f"Duplicate student IDs: {duplicates}")
        
        # Cross-dataset validation
        if 'students' in datasets and 'learning_analytics' in datasets:
            student_ids = set(datasets['students']['student_id'])
            analytics_ids = set(datasets['learning_analytics']['student_id'])
            
            missing_analytics = student_ids - analytics_ids
            if missing_analytics:
                issues['learning_analytics'].append(
                    f"Missing analytics for {len(missing_analytics)} students"
                )
        
        return {k: v for k, v in issues.items() if v}  # Return only datasets with issues
    
    @staticmethod
    def create_summary_statistics(data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Create comprehensive summary statistics"""
        summary = data[numeric_cols].describe()
        
        # Add additional statistics
        summary.loc['cv'] = data[numeric_cols].std() / data[numeric_cols].mean()
        summary.loc['skewness'] = data[numeric_cols].skew()
        summary.loc['kurtosis'] = data[numeric_cols].kurtosis()
        
        return summary.round(3)
    
    @staticmethod
    def create_correlation_matrix(data: pd.DataFrame, 
                                method: str = 'pearson',
                                threshold: float = 0.3) -> pd.DataFrame:
        """Create correlation matrix with significance filtering"""
        corr_matrix = data.corr(method=method)
        
        # Create mask for low correlations
        mask = np.abs(corr_matrix) < threshold
        corr_matrix[mask] = np.nan
        
        return corr_matrix
    
    @staticmethod
    def aggregate_by_time_window(data: pd.DataFrame, 
                               time_col: str,
                               window: str = 'D',
                               agg_dict: Optional[Dict] = None) -> pd.DataFrame:
        """Aggregate data by time windows"""
        data = data.copy()
        data[time_col] = pd.to_datetime(data[time_col])
        
        if agg_dict is None:
            # Default aggregation for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            agg_dict = {col: 'mean' for col in numeric_cols if col != time_col}
        
        return data.set_index(time_col).resample(window).agg(agg_dict)
    
    @staticmethod
    def calculate_growth_metrics(data: pd.DataFrame,
                               student_col: str,
                               metric_col: str,
                               time_col: str) -> pd.DataFrame:
        """Calculate growth metrics for each student"""
        growth_metrics = []
        
        for student in data[student_col].unique():
            student_data = data[data[student_col] == student].sort_values(time_col)
            
            if len(student_data) > 1:
                first_value = student_data[metric_col].iloc[0]
                last_value = student_data[metric_col].iloc[-1]
                
                growth = {
                    'student_id': student,
                    'initial_value': first_value,
                    'final_value': last_value,
                    'absolute_growth': last_value - first_value,
                    'relative_growth': (last_value - first_value) / first_value if first_value != 0 else 0,
                    'days_tracked': (student_data[time_col].max() - student_data[time_col].min()).days,
                    'volatility': student_data[metric_col].std(),
                    'trend_slope': np.polyfit(range(len(student_data)), student_data[metric_col], 1)[0]
                }
                growth_metrics.append(growth)
        
        return pd.DataFrame(growth_metrics)


class StreamlitHelpers:
    """Streamlit-specific helper functions"""
    
    @staticmethod
    def create_metric_card(label: str, value: float, delta: Optional[float] = None,
                          format_str: str = "{:.2f}") -> None:
        """Create a styled metric card"""
        if delta is not None:
            st.metric(label, format_str.format(value), 
                     delta=format_str.format(delta),
                     delta_color="normal" if delta >= 0 else "inverse")
        else:
            st.metric(label, format_str.format(value))
    
    @staticmethod
    def create_download_button(data: pd.DataFrame, filename: str, 
                             button_text: str = "Download CSV") -> None:
        """Create download button for dataframe"""
        csv = data.to_csv(index=False)
        st.download_button(
            label=button_text,
            data=csv,
            file_name=filename,
            mime='text/csv'
        )
    
    @staticmethod
    def create_info_box(title: str, content: str, box_type: str = "info") -> None:
        """Create styled information box"""
        box_functions = {
            "info": st.info,
            "success": st.success,
            "warning": st.warning,
            "error": st.error
        }
        
        box_func = box_functions.get(box_type, st.info)
        box_func(f"**{title}**\n\n{content}")
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def load_cached_data(file_path: str) -> pd.DataFrame:
        """Load data with caching"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    
    @staticmethod
    def create_sidebar_filters(data: pd.DataFrame) -> Dict:
        """Create standard sidebar filters"""
        filters = {}
        
        st.sidebar.header("Filters")
        
        # Date range filter
        if 'timestamp' in data.columns:
            date_col = pd.to_datetime(data['timestamp']).dt.date
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(date_col.min(), date_col.max()),
                min_value=date_col.min(),
                max_value=date_col.max()
            )
            filters['date_range'] = date_range
        
        # Student filter
        if 'student_id' in data.columns:
            students = ['All'] + sorted(data['student_id'].unique().tolist())
            selected_student = st.sidebar.selectbox("Student", students)
            filters['student'] = None if selected_student == 'All' else selected_student
        
        # Grade filter
        if 'grade_level' in data.columns:
            grades = ['All'] + sorted(data['grade_level'].unique().tolist())
            selected_grade = st.sidebar.multiselect(
                "Grade Level", 
                grades[1:],  # Exclude 'All'
                default=grades[1:]
            )
            filters['grades'] = selected_grade
        
        return filters


class AnalyticsHelpers:
    """Helper functions for analytics calculations"""
    
    @staticmethod
    def calculate_engagement_index(data: pd.DataFrame) -> pd.Series:
        """Calculate composite engagement index"""
        # Normalize metrics to 0-1 scale
        metrics = ['quest_completion_rate', 'days_active', 'collaboration_events', 
                  'building_complexity_avg', 'skill_progression']
        
        normalized = pd.DataFrame()
        for metric in metrics:
            if metric in data.columns:
                min_val = data[metric].min()
                max_val = data[metric].max()
                if max_val > min_val:
                    normalized[metric] = (data[metric] - min_val) / (max_val - min_val)
                else:
                    normalized[metric] = 0.5
        
        # Weighted average
        weights = {'quest_completion_rate': 0.3, 'days_active': 0.2, 
                  'collaboration_events': 0.2, 'building_complexity_avg': 0.15,
                  'skill_progression': 0.15}
        
        engagement_index = sum(normalized[col] * weights.get(col, 0.2) 
                              for col in normalized.columns)
        
        return engagement_index
    
    @staticmethod
    def identify_at_risk_students(data: pd.DataFrame, 
                                thresholds: Optional[Dict] = None) -> pd.DataFrame:
        """Identify at-risk students based on multiple criteria"""
        if thresholds is None:
            thresholds = {
                'engagement_score': 0.3,
                'quest_completion_rate': 0.4,
                'days_active': 5,
                'skill_progression': 0.2
            }
        
        at_risk_flags = pd.DataFrame(index=data.index)
        
        for metric, threshold in thresholds.items():
            if metric in data.columns:
                if metric == 'days_active':
                    at_risk_flags[f'{metric}_risk'] = data[metric] < threshold
                else:
                    at_risk_flags[f'{metric}_risk'] = data[metric] < threshold
        
        # Overall risk score
        at_risk_flags['risk_score'] = at_risk_flags.mean(axis=1)
        at_risk_flags['risk_level'] = pd.cut(
            at_risk_flags['risk_score'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Add student info
        result = pd.concat([
            data[['student_id', 'engagement_score', 'quest_completion_rate', 
                  'days_active', 'skill_progression']],
            at_risk_flags
        ], axis=1)
        
        return result.sort_values('risk_score', ascending=False)
    
    @staticmethod
    def calculate_peer_influence(collaboration_data: pd.DataFrame,
                               student_metrics: pd.DataFrame) -> pd.DataFrame:
        """Calculate peer influence metrics"""
        import networkx as nx
        
        # Build collaboration network
        G = nx.Graph()
        
        for _, row in collaboration_data.iterrows():
            G.add_edge(row['student_1'], row['student_2'], 
                      weight=row.get('effectiveness', 1))
        
        # Calculate network metrics
        influence_metrics = []
        
        for student in G.nodes():
            # Get student's performance
            if student in student_metrics['student_id'].values:
                student_perf = student_metrics[
                    student_metrics['student_id'] == student
                ]['engagement_score'].iloc[0]
                
                # Calculate peer average
                peers = list(G.neighbors(student))
                if peers:
                    peer_perfs = student_metrics[
                        student_metrics['student_id'].isin(peers)
                    ]['engagement_score'].mean()
                    
                    influence = {
                        'student_id': student,
                        'degree_centrality': nx.degree_centrality(G)[student],
                        'betweenness_centrality': nx.betweenness_centrality(G)[student],
                        'clustering_coefficient': nx.clustering(G, student),
                        'own_performance': student_perf,
                        'peer_avg_performance': peer_perfs,
                        'peer_influence': peer_perfs - student_perf,
                        'n_connections': len(peers)
                    }
                    influence_metrics.append(influence)
        
        return pd.DataFrame(influence_metrics)


# Configuration management
def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.warning(f"Config file {config_path} not found. Using defaults.")
        return get_default_config()

def get_default_config() -> Dict:
    """Get default configuration"""
    return {
        'simulation': {
            'n_students': 60,
            'days': 30,
            'seed': 42
        },
        'zones': {
            'spawn': {'x': 0, 'z': 0, 'radius': 30},
            'tutorial': {'x': 50, 'z': 0, 'radius': 40},
            'building_area': {'x': 100, 'z': 100, 'radius': 100},
            'collaboration_space': {'x': -100, 'z': 100, 'radius': 80},
            'challenge_zone': {'x': 0, 'z': 200, 'radius': 60},
            'resource_area': {'x': -150, 'z': -50, 'radius': 50},
            'showcase_area': {'x': 150, 'z': -50, 'radius': 70}
        },
        'analytics': {
            'engagement_weights': {
                'quest_completion': 0.3,
                'building_activity': 0.3,
                'collaboration': 0.2,
                'skill_progression': 0.2
            },
            'risk_thresholds': {
                'engagement_score': 0.3,
                'quest_completion_rate': 0.4,
                'days_active': 5
            }
        },
        'visualization': {
            'theme': 'plotly',
            'height': 600,
            'width': 1200
        }
    }

def save_config(config: Dict, config_path: str = "config.yaml") -> None:
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
```