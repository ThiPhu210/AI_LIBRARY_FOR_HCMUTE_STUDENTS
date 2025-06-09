import os
import django
import sys

# Get the absolute path of the project root directory
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_path))))
sys.path.insert(0, project_root)

# Set up Django environment
os.environ['DJANGO_SETTINGS_MODULE'] = 'SmartLibrary.settings'
django.setup()

from django.core.management.base import BaseCommand
from bookstore.models import User, StudentProfile
import cv2
import numpy as np
from django.conf import settings

class Command(BaseCommand):
    help = 'Create users from images in the specified directory'

    def add_arguments(self, parser):
        parser.add_argument('image_dir', type=str, help='Directory containing user images')
        parser.add_argument('--prefix', type=str, default='user', help='Prefix for username')

    def extract_face_features(self, image_path):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not load image")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            raise Exception("No face detected")

        # Get the first face
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        # Resize to standard size
        face_roi = cv2.resize(face_roi, (100, 100))

        # Normalize
        face_roi = face_roi.astype(np.float32) / 255.0

        # Flatten to get feature vector
        face_features = face_roi.flatten()

        return face_features

    def parse_directory_name(self, dir_name):
        """Parse directory name to extract username and id"""
        # Split by underscore
        parts = dir_name.split('_')
        
        # Last part is the ID
        student_id = parts[-1]
        
        # All parts except the last one form the name
        name_parts = parts[:-1]
        
        # Join name parts and convert to lowercase for username
        username = ''.join(name_parts).lower()
        
        # Join name parts with spaces for display name
        display_name = ' '.join(name_parts)
        
        return username, student_id, display_name

    def handle(self, *args, **options):
        base_dir = options['image_dir']
        prefix = options['prefix']

        # Create directory if it doesn't exist
        if not os.path.exists(base_dir):
            self.stdout.write(self.style.ERROR(f'Directory {base_dir} does not exist'))
            return

        # Get all subdirectories (each representing a user)
        user_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        if not user_dirs:
            self.stdout.write(self.style.ERROR(f'No user directories found in {base_dir}'))
            return

        # Process each user directory
        for user_dir in user_dirs:
            try:
                # Parse directory name to get username and id
                username, student_id, display_name = self.parse_directory_name(user_dir)
                
                # Check if user already exists
                if User.objects.filter(username=username).exists():
                    self.stdout.write(self.style.WARNING(f'User {username} already exists, skipping...'))
                    continue

                # Get all image files in the user's directory
                user_path = os.path.join(base_dir, user_dir)
                image_files = [f for f in os.listdir(user_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not image_files:
                    self.stdout.write(self.style.WARNING(f'No images found in {user_dir}, skipping...'))
                    continue

                # Create user
                user = User.objects.create_user(
                    username=username,
                    password='123456',  # Default password
                    first_name=display_name,  # Use formatted name for display
                )

                # Create student profile
                StudentProfile.objects.create(
                    user=user,
                    student_id=student_id
                )

                self.stdout.write(self.style.SUCCESS(f'Successfully created user {username} (ID: {student_id})'))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error processing {user_dir}: {str(e)}'))

        self.stdout.write(self.style.SUCCESS('Finished processing all users')) 