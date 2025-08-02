# src/data_generation/simulator.py
"""Minimal working simulator for Minecraft Education Dashboard"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

class MinecraftEducationSimulator:
    """Generates synthetic Minecraft Education data"""
    
    def __init__(self):
        self.fake = Faker()
        self.rng = np.random.RandomState(42)
    
    def generate_complete_dataset(self, n_students=60, n_days=30):
        """Generate all datasets"""
        
        # Generate students
        students = self.generate_students(n_students)
        
        # Generate other data
        movements = self.generate_movements(students, n_days)
        buildings = self.generate_buildings(students, n_days)
        quests = self.generate_quests(students, n_days)
        collaborations = self.generate_collaborations(students, n_days)
        learning_analytics = self.generate_analytics(students)
        
        return {
            'students': students,
            'movements': movements,
            'buildings': buildings,
            'quests': quests,
            'collaborations': collaborations,
            'learning_analytics': learning_analytics
        }
    
    def generate_students(self, n_students):
        """Generate student profiles"""
        data = []
        for i in range(n_students):
            data.append({
                'student_id': f'STU_{i+1:04d}',
                'username': self.fake.user_name(),
                'grade_level': np.random.choice([6, 7, 8]),
                'learning_style': np.random.choice(['visual', 'kinesthetic', 'auditory']),
                'prior_minecraft_experience': np.random.choice(['none', 'beginner', 'intermediate', 'advanced']),
                'collaboration_preference': np.random.choice(['solo', 'pairs', 'groups']),
                'stem_interest_pre': np.random.randint(1, 6)
            })
        return pd.DataFrame(data)
    
    def generate_movements(self, students, n_days):
        """Generate movement data"""
        data = []
        zones = ['spawn', 'tutorial', 'building_area', 'collaboration_space', 'challenge_zone']
        
        for _, student in students.iterrows():
            for day in range(n_days):
                for _ in range(np.random.randint(10, 50)):
                    timestamp = datetime.now() - timedelta(days=n_days-day, hours=np.random.randint(0, 24))
                    data.append({
                        'student_id': student['student_id'],
                        'timestamp': timestamp,
                        'x': np.random.uniform(-200, 200),
                        'y': np.random.uniform(60, 100),
                        'z': np.random.uniform(-200, 200),
                        'zone': np.random.choice(zones),
                        'session_id': f"{student['student_id']}_{day}"
                    })
        return pd.DataFrame(data)
    
    def generate_buildings(self, students, n_days):
        """Generate building data"""
        data = []
        block_types = ['wood', 'stone', 'glass', 'iron', 'redstone']
        
        for _, student in students.iterrows():
            for day in range(n_days):
                for _ in range(np.random.randint(5, 30)):
                    timestamp = datetime.now() - timedelta(days=n_days-day)
                    data.append({
                        'student_id': student['student_id'],
                        'timestamp': timestamp,
                        'action': np.random.choice(['place', 'break']),
                        'block_type': np.random.choice(block_types),
                        'x': np.random.uniform(-200, 200),
                        'y': np.random.uniform(60, 100),
                        'z': np.random.uniform(-200, 200),
                        'structure_id': f"STRUCT_{day}",
                        'complexity_score': np.random.uniform(1, 10)
                    })
        return pd.DataFrame(data)
    
    def generate_quests(self, students, n_days):
        """Generate quest data"""
        data = []
        quest_names = ['Basic Building', 'Redstone Basics', 'Collaborative Castle', 'Resource Management']
        
        for _, student in students.iterrows():
            for day in range(n_days):
                for quest in quest_names:
                    if np.random.random() > 0.5:
                        timestamp = datetime.now() - timedelta(days=n_days-day)
                        completed = np.random.random() > 0.3
                        data.append({
                            'student_id': student['student_id'],
                            'quest_name': quest,
                            'start_time': timestamp,
                            'completion_time': timestamp + timedelta(minutes=np.random.randint(10, 60)),
                            'completed': completed,
                            'attempts': np.random.randint(1, 5),
                            'score': np.random.randint(60, 100) if completed else np.random.randint(0, 60),
                            'difficulty': np.random.randint(1, 5)
                        })
        return pd.DataFrame(data)
    
    def generate_collaborations(self, students, n_days):
        """Generate collaboration data"""
        data = []
        interaction_types = ['chat', 'build_together', 'resource_share', 'help']
        
        student_ids = students['student_id'].tolist()
        for day in range(n_days):
            for _ in range(np.random.randint(10, 50)):
                s1, s2 = np.random.choice(student_ids, 2, replace=False)
                timestamp = datetime.now() - timedelta(days=n_days-day)
                data.append({
                    'timestamp': timestamp,
                    'student_1': s1,
                    'student_2': s2,
                    'interaction_type': np.random.choice(interaction_types),
                    'duration_minutes': np.random.randint(1, 30),
                    'zone': np.random.choice(['collaboration_space', 'building_area']),
                    'effectiveness': np.random.uniform(0, 1)
                })
        return pd.DataFrame(data)
    
    def generate_analytics(self, students):
        """Generate learning analytics"""
        data = []
        for _, student in students.iterrows():
            engagement = np.random.uniform(0.3, 1.0)
            data.append({
                'student_id': student['student_id'],
                'engagement_score': engagement,
                'quest_completion_rate': np.random.uniform(0.4, 1.0),
                'avg_attempts_per_quest': np.random.uniform(1, 4),
                'building_complexity_avg': np.random.uniform(2, 8),
                'total_blocks_placed': np.random.randint(100, 5000),
                'collaboration_events': np.random.randint(5, 100),
                'days_active': np.random.randint(5, 30),
                'skill_progression': np.random.uniform(0.2, 1.0),
                'stem_interest_pre': student['stem_interest_pre'],
                'stem_interest_post': min(5, student['stem_interest_pre'] + np.random.randint(0, 3)),
                'learning_gain': np.random.uniform(-0.5, 2.0)
            })
        return pd.DataFrame(data)

def create_config():
    """Create configuration file"""
    import yaml
    config = {
        'simulation': {'n_students': 60, 'days': 30, 'seed': 42},
        'zones': {
            'spawn': {'x': 0, 'z': 0, 'radius': 30},
            'tutorial': {'x': 50, 'z': 0, 'radius': 40}
        }
    }
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)
