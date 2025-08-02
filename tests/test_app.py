### test_app.py
```python
import unittest
import pandas as pd
import numpy as np
from src.data_generation.simulator import MinecraftEducationSimulator
from src.analysis.statistical import EducationalStatisticsAnalyzer
from src.analysis.time_series import TimeSeriesEducationAnalyzer

class TestMinecraftEducationDashboard(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = MinecraftEducationSimulator()
        self.stat_analyzer = EducationalStatisticsAnalyzer()
        self.ts_analyzer = TimeSeriesEducationAnalyzer()
        
        # Generate small test dataset
        self.test_data = self.simulator.generate_complete_dataset(
            n_students=10, days=7
        )
    
    def test_data_generation(self):
        """Test data generation integrity"""
        # Check all required datasets are generated
        required_datasets = ['students', 'movements', 'buildings', 
                           'quests', 'collaborations', 'learning_analytics']
        
        for dataset in required_datasets:
            self.assertIn(dataset, self.test_data)
            self.assertGreater(len(self.test_data[dataset]), 0)
        
        # Check student count
        self.assertEqual(len(self.test_data['students']), 10)
    
    def test_statistical_analysis(self):
        """Test statistical analysis functions"""
        # Create test groups
        test_df = pd.DataFrame({
            'group': ['A'] * 20 + ['B'] * 20,
            'outcome': np.random.normal(0.5, 0.1, 20).tolist() + 
                      np.random.normal(0.7, 0.1, 20).tolist()
        })
        
        # Run t-test
        results = self.stat_analyzer.compare_learning_methods(
            test_df, 'group', 'outcome'
        )
        
        # Check results structure
        self.assertIn('p_value', results)
        self.assertIn('effect_size', results)
        self.assertIn('interpretation', results)
    
    def test_time_series_analysis(self):
        """Test time series analysis"""
        # Create test time series
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        ts_data = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(30).cumsum() + 10
        })
        
        # Run analysis
        results = self.ts_analyzer.analyze_engagement_patterns(
            ts_data, 'timestamp', 'value'
        )
        
        # Check results
        self.assertIn('summary_statistics', results)
        self.assertIn('trend', results)
        self.assertIn('forecast', results)
    
    def test_changepoint_detection(self):
        """Test changepoint detection"""
        # Create data with clear changepoint
        values1 = np.random.normal(5, 1, 50)
        values2 = np.random.normal(10, 1, 50)
        values = np.concatenate([values1, values2])
        
        test_df = pd.DataFrame({
            'metric': values
        })
        
        # Detect changepoints
        results = self.ts_analyzer.detect_learning_changepoints(
            test_df, 'metric'
        )
        
        # Check results
        self.assertIn('segments', results)
        self.assertIn('learning_phases', results)

if __name__ == '__main__':
    unittest.main()
```