from django.core.management.base import BaseCommand
from bookstore.recommend.trainer import RLTrainer
import torch

class Command(BaseCommand):
    help = 'Train recommendation model with historical data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--episodes',
            type=int,
            default=70,
            help='Number of training episodes (default: 70)'
        )
        parser.add_argument(
            '--gamma',
            type=float,
            default=0.99,
            help='Discount factor for rewards (default: 0.99)'
        )
        parser.add_argument(
            '--patience',
            type=int,
            default=50,
            help='Early stopping patience (default: 50)'
        )
        parser.add_argument(
            '--device',
            type=str,
            default='cpu',
            choices=['cpu', 'cuda'],
            help='Device to train on (default: cpu)'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=1e-5,
            help='Learning rate for optimizer (default: 1e-5)'
        )
        parser.add_argument(
            '--weight-decay',
            type=float,
            default=1e-4,
            help='Weight decay for optimizer (default: 1e-4)'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting recommendation model training...'))
        
        # Kiểm tra CUDA availability
        if options['device'] == 'cuda' and not torch.cuda.is_available():
            self.stdout.write(self.style.WARNING('CUDA requested but not available. Falling back to CPU.'))
            options['device'] = 'cpu'
        
        self.stdout.write(f"Training on device: {options['device']}")
        self.stdout.write(f"Number of episodes: {options['episodes']}")
        self.stdout.write(f"Gamma: {options['gamma']}")
        self.stdout.write(f"Patience: {options['patience']}")
        self.stdout.write(f"Learning rate: {options['learning_rate']}")
        self.stdout.write(f"Weight decay: {options['weight_decay']}")
        
        # Khởi tạo trainer với device
        trainer = RLTrainer(device=options['device'])
        
        # Train model với tất cả tham số
        trainer.train(
            episodes=options['episodes'],
            gamma=options['gamma'],
            patience=options['patience'],
            learning_rate=options['learning_rate'],
            weight_decay=options['weight_decay']
        )
        
        self.stdout.write(self.style.SUCCESS('Training completed successfully!'))
        self.stdout.write('Model saved to: bookstore/recommend/model.pt')
        self.stdout.write('Training curve saved to: reward_curve.png') 