## Step 2: Data Generation Module

### src/data_generation/simulator.py
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from faker import Faker
import networkx as nx
from typing import Dict, List, Tuple
import yaml

class MinecraftEducationSimulator:
    """Generates realistic Minecraft Education gameplay data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.fake = Faker()
        self.rng = np.random.RandomState(42)
        
    def generate_students(self, n_students: int = 30) -> pd.DataFrame:
        """Generate student profiles with realistic demographics"""
        
        students = []
        for i in range(n_students):
            student = {
                'student_id': f"STU_{i+1:04d}",
                'username': self.fake.user_name(),
                'grade_level': self.rng.choice([6, 7, 8], p=[0.33, 0.34, 0.33]),
                'learning_style': self.rng.choice(['visual', 'kinesthetic', 'auditory'], 
                                                p=[0.4, 0.4, 0.2]),
                'prior_minecraft_experience': self.rng.choice(['none', 'beginner', 'intermediate', 'advanced'],
                                                            p=[0.3, 0.4, 0.25, 0.05]),
                'collaboration_preference': self.rng.choice(['solo', 'pairs', 'groups'],
                                                          p=[0.25, 0.45, 0.3]),
                'stem_interest_pre': self.rng.randint(1, 6),  # 1-5 Likert scale
                'created_at': datetime.now() - timedelta(days=self.rng.randint(0, 180))
            }
            students.append(student)
            
        return pd.DataFrame(students)
    
    def generate_movement_data(self, students_df: pd.DataFrame, 
                             start_date: datetime, days: int = 30) -> pd.DataFrame:
        """Generate realistic player movement patterns"""
        
        movements = []
        zones = ['spawn', 'tutorial', 'building_area', 'collaboration_space', 
                 'challenge_zone', 'resource_area', 'showcase_area']
        
        for _, student in students_df.iterrows():
            # Student-specific parameters
            activity_level = self._get_activity_level(student['prior_minecraft_experience'])
            
            for day in range(days):
                if self.rng.random() > 0.3:  # 70% attendance
                    session_date = start_date + timedelta(days=day)
                    session_duration = self.rng.lognormal(3.5, 0.5)  # Log-normal session length
                    session_duration = min(max(session_duration, 10), 120)  # 10-120 minutes
                    
                    # Generate movements within session
                    n_movements = int(session_duration * activity_level)
                    current_zone = 'spawn'
                    
                    for minute in range(int(session_duration)):
                        if self.rng.random() < 0.15:  # 15% chance to change zone
                            current_zone = self._next_zone(current_zone, zones, student)
                        
                        # Generate position within zone
                        x = self.rng.normal(self._zone_center(current_zone)[0], 20)
                        y = self.rng.randint(60, 100)
                        z = self.rng.normal(self._zone_center(current_zone)[1], 20)
                        
                        movement = {
                            'student_id': student['student_id'],
                            'timestamp': session_date + timedelta(minutes=minute),
                            'x': x, 'y': y, 'z': z,
                            'zone': current_zone,
                            'session_id': f"{student['student_id']}_{session_date.strftime('%Y%m%d')}",
                            'minute_in_session': minute
                        }
                        movements.append(movement)
        
        return pd.DataFrame(movements)
    
    def generate_building_data(self, movements_df: pd.DataFrame) -> pd.DataFrame:
        """Generate block placement and building data"""
        
        buildings = []
        block_types = ['wood', 'stone', 'glass', 'iron', 'redstone', 'wool', 'concrete']
        
        for session_id in movements_df['session_id'].unique():
            session_data = movements_df[movements_df['session_id'] == session_id]
            student_id = session_data['student_id'].iloc[0]
            
            # Building events based on zone
            building_zones = session_data[session_data['zone'].isin(['building_area', 'challenge_zone'])]
            
            for _, movement in building_zones.iterrows():
                if self.rng.random() < 0.3:  # 30% chance of building action
                    action_type = self.rng.choice(['place', 'break'], p=[0.8, 0.2])
                    
                    building = {
                        'student_id': student_id,
                        'timestamp': movement['timestamp'],
                        'action': action_type,
                        'block_type': self.rng.choice(block_types),
                        'x': movement['x'] + self.rng.randint(-5, 5),
                        'y': movement['y'] + self.rng.randint(-2, 10),
                        'z': movement['z'] + self.rng.randint(-5, 5),
                        'structure_id': f"STRUCT_{session_id}_{self.rng.randint(1, 10)}",
                        'complexity_score': self.rng.lognormal(2, 0.5)  # Building complexity
                    }
                    buildings.append(building)
        
        return pd.DataFrame(buildings)
    
    def generate_quest_data(self, students_df: pd.DataFrame, 
                           start_date: datetime, days: int = 30) -> pd.DataFrame:
        """Generate quest completion and learning progression data"""
        
        quests = []
        quest_types = [
            {'name': 'Basic Building', 'difficulty': 1, 'skills': ['construction']},
            {'name': 'Redstone Basics', 'difficulty': 2, 'skills': ['logic', 'engineering']},
            {'name': 'Collaborative Castle', 'difficulty': 3, 'skills': ['teamwork', 'planning']},
            {'name': 'Resource Management', 'difficulty': 2, 'skills': ['economics', 'planning']},
            {'name': 'Science Lab', 'difficulty': 4, 'skills': ['science', 'experimentation']},
            {'name': 'Math Puzzles', 'difficulty': 3, 'skills': ['mathematics', 'problem_solving']}
        ]
        
        for _, student in students_df.iterrows():
            skill_level = self._initial_skill_level(student)
            
            for day in range(days):
                if self.rng.random() > 0.4:  # 60% chance of quest activity
                    date = start_date + timedelta(days=day)
                    
                    # Select appropriate quest based on skill level
                    available_quests = [q for q in quest_types 
                                      if abs(q['difficulty'] - skill_level) <= 1.5]
                    
                    if available_quests:
                        quest = self.rng.choice(available_quests)
                        
                        # Calculate completion based on skill match
                        base_success = 0.3 + (0.2 * skill_level / 5)
                        success_prob = min(0.9, base_success + (0.1 * student['grade_level'] / 8))
                        
                        attempts = self.rng.geometric(success_prob)
                        completed = attempts <= 3
                        
                        # Update skill level based on completion
                        if completed:
                            skill_level = min(5, skill_level + 0.1 * quest['difficulty'])
                        
                        quest_record = {
                            'student_id': student['student_id'],
                            'quest_name': quest['name'],
                            'start_time': date,
                            'completion_time': date + timedelta(minutes=self.rng.randint(10, 60)),
                            'completed': completed,
                            'attempts': attempts,
                            'score': self.rng.randint(60, 100) if completed else self.rng.randint(0, 59),
                            'skills_developed': ','.join(quest['skills']),
                            'difficulty': quest['difficulty']
                        }
                        quests.append(quest_record)
        
        return pd.DataFrame(quests)
    
    def generate_collaboration_data(self, students_df: pd.DataFrame,
                                  movements_df: pd.DataFrame) -> pd.DataFrame:
        """Generate peer interaction and collaboration events"""
        
        collaborations = []
        
        # Create collaboration network
        G = nx.erdos_renyi_graph(len(students_df), 0.3, seed=42)
        
        # Group sessions by timestamp to find overlaps
        movements_df['timestamp_rounded'] = pd.to_datetime(movements_df['timestamp']).dt.floor('5min')
        
        for timestamp, group in movements_df.groupby('timestamp_rounded'):
            if len(group) > 1:
                # Students in same zone at same time
                zone_groups = group.groupby('zone')
                
                for zone, students_in_zone in zone_groups:
                    if len(students_in_zone) > 1 and zone in ['collaboration_space', 'building_area']:
                        student_ids = students_in_zone['student_id'].unique()
                        
                        # Generate pairwise interactions
                        for i in range(len(student_ids)):
                            for j in range(i+1, len(student_ids)):
                                if self.rng.random() < 0.2:  # 20% interaction probability
                                    collab_type = self.rng.choice(['chat', 'build_together', 
                                                                 'resource_share', 'help'])
                                    
                                    collaboration = {
                                        'timestamp': timestamp,
                                        'student_1': student_ids[i],
                                        'student_2': student_ids[j],
                                        'interaction_type': collab_type,
                                        'duration_minutes': self.rng.randint(1, 15),
                                        'zone': zone,
                                        'effectiveness': self.rng.beta(2, 1)  # Skewed toward effective
                                    }
                                    collaborations.append(collaboration)
        
        return pd.DataFrame(collaborations)
    
    def generate_complete_dataset(self, n_students: int = 30, days: int = 30) -> Dict[str, pd.DataFrame]:
        """Generate complete synthetic dataset"""
        
        print("Generating student profiles...")
        students = self.generate_students(n_students)
        
        print("Generating movement data...")
        start_date = datetime.now() - timedelta(days=days)
        movements = self.generate_movement_data(students, start_date, days)
        
        print("Generating building data...")
        buildings = self.generate_building_data(movements)
        
        print("Generating quest data...")
        quests = self.generate_quest_data(students, start_date, days)
        
        print("Generating collaboration data...")
        collaborations = self.generate_collaboration_data(students, movements)
        
        # Calculate derived metrics
        print("Calculating learning analytics...")
        learning_analytics = self._calculate_learning_metrics(students, quests, buildings, collaborations)
        
        return {
            'students': students,
            'movements': movements,
            'buildings': buildings,
            'quests': quests,
            'collaborations': collaborations,
            'learning_analytics': learning_analytics
        }
    
    # Helper methods
    def _get_activity_level(self, experience: str) -> float:
        """Map experience to activity level"""
        mapping = {'none': 0.5, 'beginner': 0.8, 'intermediate': 1.2, 'advanced': 1.5}
        return mapping.get(experience, 1.0)
    
    def _zone_center(self, zone: str) -> Tuple[float, float]:
        """Get x,z coordinates for zone centers"""
        centers = {
            'spawn': (0, 0),
            'tutorial': (50, 0),
            'building_area': (100, 100),
            'collaboration_space': (-100, 100),
            'challenge_zone': (0, 200),
            'resource_area': (-150, -50),
            'showcase_area': (150, -50)
        }
        return centers.get(zone, (0, 0))
    
    def _next_zone(self, current: str, zones: List[str], student: pd.Series) -> str:
        """Determine next zone based on student preferences"""
        if student['collaboration_preference'] == 'groups' and self.rng.random() < 0.4:
            return 'collaboration_space'
        return self.rng.choice(zones)
    
    def _initial_skill_level(self, student: pd.Series) -> float:
        """Calculate initial skill level"""
        base = {'none': 1, 'beginner': 2, 'intermediate': 3, 'advanced': 4}
        return base.get(student['prior_minecraft_experience'], 2) + self.rng.normal(0, 0.5)
    
    def _calculate_learning_metrics(self, students: pd.DataFrame, quests: pd.DataFrame,
                                  buildings: pd.DataFrame, collaborations: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive learning analytics"""
        
        metrics = []
        
        for _, student in students.iterrows():
            student_id = student['student_id']
            
            # Quest metrics
            student_quests = quests[quests['student_id'] == student_id]
            quest_completion_rate = student_quests['completed'].mean() if len(student_quests) > 0 else 0
            avg_attempts = student_quests['attempts'].mean() if len(student_quests) > 0 else 0
            
            # Building metrics
            student_buildings = buildings[buildings['student_id'] == student_id]
            building_complexity = student_buildings['complexity_score'].mean() if len(student_buildings) > 0 else 0
            blocks_placed = len(student_buildings[student_buildings['action'] == 'place'])
            
            # Collaboration metrics
            student_collabs = collaborations[(collaborations['student_1'] == student_id) | 
                                           (collaborations['student_2'] == student_id)]
            collaboration_count = len(student_collabs)
            
            # Calculate learning progression (simulated)
            days_active = student_quests['start_time'].nunique() if len(student_quests) > 0 else 0
            learning_rate = 0.1 + (0.05 * student['grade_level'] / 8)
            skill_progression = 1 - np.exp(-learning_rate * days_active)
            
            # Engagement score
            engagement_score = (quest_completion_rate * 0.3 + 
                              min(blocks_placed / 1000, 1) * 0.3 +
                              min(collaboration_count / 50, 1) * 0.2 +
                              skill_progression * 0.2)
            
            # Predict post-STEM interest
            stem_interest_post = min(5, student['stem_interest_pre'] + 
                                   self.rng.normal(engagement_score * 2, 0.5))
            
            metric = {
                'student_id': student_id,
                'quest_completion_rate': quest_completion_rate,
                'avg_attempts_per_quest': avg_attempts,
                'building_complexity_avg': building_complexity,
                'total_blocks_placed': blocks_placed,
                'collaboration_events': collaboration_count,
                'days_active': days_active,
                'skill_progression': skill_progression,
                'engagement_score': engagement_score,
                'stem_interest_pre': student['stem_interest_pre'],
                'stem_interest_post': int(np.round(stem_interest_post)),
                'learning_gain': stem_interest_post - student['stem_interest_pre']
            }
            metrics.append(metric)
        
        return pd.DataFrame(metrics)


# Create configuration file
def create_config():
    config = {
        'simulation': {
            'n_students': 120,
            'days': 60,
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
        'quests': {
            'max_difficulty': 5,
            'completion_bonus': 100,
            'attempt_penalty': 10
        },
        'analytics': {
            'engagement_weights': {
                'quest_completion': 0.3,
                'building_activity': 0.3,
                'collaboration': 0.2,
                'skill_progression': 0.2
            }
        }
    }
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
```