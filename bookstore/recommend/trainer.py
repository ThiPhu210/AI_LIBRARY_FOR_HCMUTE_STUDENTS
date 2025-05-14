import torch
from torch import nn, optim
import random
import numpy as np
import os
import sys
import django
import matplotlib.pyplot as plt
import gc
# Thêm đường dẫn gốc project vào PYTHONPATH


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(53)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# Thiết lập Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bookapp.settings')
django.setup()

from bookstore.recommend.transformer import RecTransformer  # Giờ sẽ import được
from bookstore.recommend.embeddings import get_book_embedding
from bookstore.models import User, Book, UserBookInteraction

class RLTrainer:
    def __init__(self, device='cpu'):
        self.books = list(Book.objects.all())
        self.book_to_idx = {book.id: idx for idx, book in enumerate(self.books)}
        self.idx_to_book = {idx: book for idx, book in self.book_to_idx.items()}
        self.n_books = len(self.books)
        self.model = RecTransformer(n_books=self.n_books).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-4)
        self.device = device

    def train(self, episodes=1000, gamma=0.99, patience=50):
        users = list(User.objects.all())
        episode_rewards = []
        smoothed_rewards = []
        best_reward = -np.inf
        patience_counter = 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            torch.cuda.init()

        for episode in range(episodes):
            user = random.choice(users)
            interactions = list(
                UserBookInteraction.objects.filter(user=user).order_by('timestamp')
            )

            if len(interactions) < 3:
                continue

            seq_embeddings = [
                torch.tensor(get_book_embedding(inter.book), dtype=torch.float32)
                for inter in interactions[:-1]
            ]
            input_tensor = torch.stack(seq_embeddings).unsqueeze(1).to(self.device)

            logits = self.model(input_tensor)
            last_inter = interactions[-1]
            target_idx = self.book_to_idx.get(last_inter.book.id, None)

            if target_idx is None:
                continue

            reward = last_inter.get_reward()
            scaled_reward = reward  # hoặc thử scaled_reward = reward / max_reward nếu reward gốc lớn

            probs = torch.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            log_prob = torch.log(probs[0, target_idx] + 1e-8)
            predicted_idx = torch.argmax(probs[0], dim=0).item()
            prediction_bonus = 2.0 if predicted_idx == target_idx else 0.0  # tăng bonus

            discounted_reward = (scaled_reward + prediction_bonus) * (gamma ** (len(interactions) - 1))

            # Giảm hệ số entropy
            loss = -log_prob * discounted_reward - 0.01 * entropy

            if predicted_idx != target_idx:
                loss += 0.01

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            episode_rewards.append(discounted_reward)
            smoothed_rewards.append(np.mean(episode_rewards[-50:]))

            if smoothed_rewards[-1] > best_reward:
                best_reward = smoothed_rewards[-1]
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(os.path.dirname(__file__), 'model.pt'))
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print(f"Early stopping at episode {episode}")
                    break

            if episode % 10 == 0:
                print(f"Episode {episode}: Loss={loss.item():.4f}, Reward={smoothed_rewards[-1]:.4f}")

        # Vẽ biểu đồ reward
        plt.figure(figsize=(6, 4))
        plt.plot(episode_rewards, alpha=0.3, label='Raw Reward')
        plt.plot(smoothed_rewards, label='Smoothed Reward')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Max Normalized Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.xlim(0, 70)  # Giới hạn trục x từ 0 đến 0.3
        plt.ylim(0, 0.30)
        plt.legend()
        plt.savefig('reward_curve.png')

        window_size = 100000000
        smoothed_ma = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smoothed_ma, label=f'MA ({window_size} episodes)')

        # Lưu model
        save_path = os.path.join(os.path.dirname(__file__), 'model.pt')
        torch.save(self.model.state_dict(), save_path)



if __name__ == "__main__":
    device = 'cpu'
    print(f"Using device: {device}")
    
    trainer = RLTrainer(device=device)
    trainer.train(episodes=70)