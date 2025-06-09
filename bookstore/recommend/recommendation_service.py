import torch
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import os
from django.core.cache import cache
from bookstore.models import User, Book, UserBookInteraction
from .transformer import RecTransformer
from .embeddings import get_book_embedding

class RecommendationService:
    def __init__(self, device='cpu'):
        self.device = device
        self.books = list(Book.objects.all())
        self.book_to_idx = {book.id: idx for idx, book in enumerate(self.books)}
        self.idx_to_book = {idx: book for idx, book in self.book_to_idx.items()}
        self.n_books = len(self.books)
        self.model = RecTransformer(n_books=self.n_books).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5, weight_decay=1e-5)
        self.interaction_queue = deque(maxlen=1000)  # Queue lưu tối đa 1000 tương tác
        self.batch_size = 100  # Kích thước batch để update model
        self.cache_timeout = 3600  # Cache timeout: 1 giờ
        self.load_model()

    def load_model(self):
        """Load model từ file đã lưu"""
        model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Model loaded successfully")

    def save_model(self):
        """Lưu model hiện tại"""
        model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
        torch.save(self.model.state_dict(), model_path)
        print("Model saved successfully")

    def get_recommendations(self, user_id, n_recommendations=10):
        """Lấy recommendations cho user"""
        cache_key = f'user_recommendations_{user_id}'
        
        # Kiểm tra cache
        if cached := cache.get(cache_key):
            return cached

        # Lấy tương tác của user
        interactions = list(UserBookInteraction.objects.filter(
            user_id=user_id
        ).order_by('timestamp'))

        if len(interactions) < 3:
            return []

        # Tạo input tensor
        seq_embeddings = [
            torch.tensor(get_book_embedding(inter.book), dtype=torch.float32)
            for inter in interactions
        ]
        input_tensor = torch.stack(seq_embeddings).unsqueeze(1).to(self.device)

        # Dự đoán
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits[-1], dim=0)
            
        # Lấy top N recommendations
        top_probs, top_indices = torch.topk(probs, n_recommendations)
        recommendations = [
            {
                'book_id': self.idx_to_book[idx.item()],
                'score': prob.item()
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

        # Cache kết quả
        cache.set(cache_key, recommendations, self.cache_timeout)
        return recommendations

    def add_interaction(self, interaction):
        """Thêm tương tác mới vào queue"""
        self.interaction_queue.append(interaction)
        
        # Nếu đủ batch_size, update model
        if len(self.interaction_queue) >= self.batch_size:
            self.update_model()

    def update_model(self):
        """Update model với các tương tác mới"""
        if not self.interaction_queue:
            return

        print(f"Updating model with {len(self.interaction_queue)} new interactions")
        
        # Lấy batch tương tác
        batch = list(self.interaction_queue)
        self.interaction_queue.clear()

        total_reward = 0
        for interaction in batch:
            # Lấy sequence của user
            user_interactions = list(UserBookInteraction.objects.filter(
                user=interaction.user
            ).order_by('timestamp'))

            if len(user_interactions) < 3:
                continue

            # Train với sequence này
            reward = self.train_single_episode(user_interactions)
            total_reward += reward

        # Lưu model sau khi update
        self.save_model()
        print(f"Model updated. Average reward: {total_reward/len(batch):.4f}")

    def train_single_episode(self, interactions):
        """Train model với một sequence tương tác"""
        seq_embeddings = [
            torch.tensor(get_book_embedding(inter.book), dtype=torch.float32)
            for inter in interactions[:-1]
        ]
        input_tensor = torch.stack(seq_embeddings).unsqueeze(1).to(self.device)

        logits = self.model(input_tensor)
        last_inter = interactions[-1]
        target_idx = self.book_to_idx.get(last_inter.book.id, None)

        if target_idx is None:
            return 0

        # Tính reward
        base_reward = last_inter.get_reward() * 2.0
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        log_prob = torch.log(probs[0, target_idx] + 1e-8)
        predicted_idx = torch.argmax(probs[0], dim=0).item()
        
        prediction_bonus = 5.0 if predicted_idx == target_idx else 0.0
        discounted_reward = (base_reward + prediction_bonus) * (0.95 ** (len(interactions) - 1))

        # Tính loss và update
        loss = -log_prob * discounted_reward - 0.005 * entropy
        if predicted_idx != target_idx:
            loss += 0.005

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return discounted_reward

# Singleton instance
recommendation_service = None

def get_recommendation_service():
    """Lấy instance của recommendation service"""
    global recommendation_service
    if recommendation_service is None:
        recommendation_service = RecommendationService()
    return recommendation_service 