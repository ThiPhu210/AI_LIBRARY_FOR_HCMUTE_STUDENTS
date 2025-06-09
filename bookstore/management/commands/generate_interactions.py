import random
from datetime import timedelta
from django.utils import timezone
from django.core.management.base import BaseCommand
from bookstore.models import User, Book, UserBookInteraction
from collections import defaultdict

class Command(BaseCommand):
    help = 'Generate 100,000 random user-book interactions'

    def handle(self, *args, **options):
        users = list(User.objects.all().values_list('id', flat=True))
        books = list(Book.objects.all().values_list('id', flat=True))
        actions = ['view', 'ask', 'download']
        now = timezone.now()

        if not users or not books:
            self.stdout.write(self.style.ERROR("❌ Không có user hoặc book trong database."))
            return

        # Tính số interaction trung bình cho mỗi user và mỗi sách
        avg_interactions_per_user = 100000 // len(users)  # Khoảng 1300 interactions/user
        avg_interactions_per_book = 100000 // len(books)  # Khoảng 400 interactions/book

        self.stdout.write(f"Tạo 100,000 interactions cho {len(users)} users và {len(books)} sách...")
        self.stdout.write(f"Trung bình {avg_interactions_per_user} interactions/user và {avg_interactions_per_book} interactions/book")

        # Theo dõi số lượng interaction của mỗi user và sách
        user_interaction_count = defaultdict(int)
        book_interaction_count = defaultdict(int)

        interactions = []
        total_created = 0

        while total_created < 100000:
            # Ưu tiên chọn user và sách có ít interaction
            available_users = [u for u in users if user_interaction_count[u] < avg_interactions_per_user * 1.5]
            available_books = [b for b in books if book_interaction_count[b] < avg_interactions_per_book * 1.5]

            if not available_users or not available_books:
                break

            user_id = random.choice(available_users)
            book_id = random.choice(available_books)
            action = random.choice(actions)
            # Chọn thời điểm trong 2 năm qua
            random_days = random.randint(0, 730)
            random_seconds = random.randint(0, 86400)
            timestamp = now - timedelta(days=random_days, seconds=random_seconds)

            interaction = UserBookInteraction(
                user_id=user_id,
                book_id=book_id,
                action=action,
                timestamp=timestamp
            )
            interactions.append(interaction)
            
            # Cập nhật số lượng interaction
            user_interaction_count[user_id] += 1
            book_interaction_count[book_id] += 1
            total_created += 1

            # Bulk insert mỗi 1000 dòng
            if len(interactions) >= 1000:
                UserBookInteraction.objects.bulk_create(interactions)
                interactions.clear()
                self.stdout.write(self.style.SUCCESS(f"✔️ Đã thêm {total_created}/100,000 interactions..."))

        # Thêm các dòng còn lại
        if interactions:
            UserBookInteraction.objects.bulk_create(interactions)

        # Hiển thị thống kê
        self.stdout.write("\nThống kê:")
        self.stdout.write(f"Tổng số interactions đã tạo: {total_created}")
        self.stdout.write(f"Trung bình interactions/user: {sum(user_interaction_count.values())/len(users):.2f}")
        self.stdout.write(f"Trung bình interactions/book: {sum(book_interaction_count.values())/len(books):.2f}")
        self.stdout.write(self.style.SUCCESS("✅ Hoàn thành!"))
