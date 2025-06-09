import os
import glob
from django.core.files import File
from django.core.management.base import BaseCommand
from bookstore.models import Book, Department

class Command(BaseCommand):
    help = 'Create book data from PDF and cover files organized by departments'

    def handle(self, *args, **options):
        # Lấy đường dẫn đến thư mục commands
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Đường dẫn đến thư mục data trong commands
        PDF_FOLDER = os.path.join(current_dir, 'data', 'pdfs')
        COVER_FOLDER = os.path.join(current_dir, 'data', 'covers')

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(PDF_FOLDER, exist_ok=True)
        os.makedirs(COVER_FOLDER, exist_ok=True)

        # Kiểm tra xem có thư mục Department nào không
        try:
            department_folders = [d for d in os.listdir(PDF_FOLDER) if os.path.isdir(os.path.join(PDF_FOLDER, d))]
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f"❌ Không tìm thấy thư mục {PDF_FOLDER}"))
            self.stdout.write(self.style.WARNING("Vui lòng tạo cấu trúc thư mục như sau:"))
            self.stdout.write(self.style.WARNING("""
data/
├── pdfs/
│   ├── Department1/
│   │   ├── book1.pdf
│   │   └── book2.pdf
│   └── Department2/
│       ├── book3.pdf
│       └── book4.pdf
└── covers/
    ├── book1.jpg
    ├── book2.jpg
    ├── book3.jpg
    └── book4.jpg
            """))
            return

        if not department_folders:
            self.stdout.write(self.style.WARNING("⚠️ Không tìm thấy thư mục Department nào trong data/pdfs/"))
            return

        for dept_folder in department_folders:
            # Tạo hoặc lấy Department
            department, created = Department.objects.get_or_create(
                name=dept_folder
            )
            
            if created:
                self.stdout.write(self.style.SUCCESS(f"✅ Đã tạo Department: {dept_folder}"))
            
            # Đường dẫn đến thư mục PDF của Department
            dept_pdf_folder = os.path.join(PDF_FOLDER, dept_folder)
            
            # Lấy danh sách file PDF trong thư mục Department
            pdf_files = [f for f in os.listdir(dept_pdf_folder) if f.endswith('.pdf')]

            if not pdf_files:
                self.stdout.write(self.style.WARNING(f"⚠️ Không tìm thấy file PDF nào trong thư mục {dept_folder}"))
                continue

            for pdf_file in pdf_files:
                # Chuyển 'Cong_nghe_so.pdf' -> 'Cong nghe so'
                title_raw = pdf_file[:-4]
                title = title_raw.replace('_', ' ').strip()

                pdf_path = os.path.join(dept_pdf_folder, pdf_file)

                # Tìm ảnh bìa có tên giống file pdf (bỏ đuôi)
                cover_pattern = os.path.join(COVER_FOLDER, f"{title_raw}.*")
                cover_matches = glob.glob(cover_pattern)

                if not cover_matches:
                    self.stdout.write(self.style.ERROR(f"❌ Không tìm thấy ảnh bìa cho: {title}"))
                    continue

                cover_path = cover_matches[0]

                # Khởi tạo book với title và department
                book = Book(
                    title=title,
                    department=department
                )

                # Gán file PDF và ảnh bìa
                try:
                    with open(pdf_path, 'rb') as f_pdf:
                        book.pdf.save(os.path.basename(pdf_path), File(f_pdf), save=False)
                except FileNotFoundError:
                    self.stdout.write(self.style.ERROR(f"❌ File not found: {pdf_path}"))
                    continue

                try:
                    with open(cover_path, 'rb') as f_img:
                        book.cover.save(os.path.basename(cover_path), File(f_img), save=False)
                except FileNotFoundError:
                    self.stdout.write(self.style.ERROR(f"❌ Cover file not found: {cover_path}"))
                    continue

                book.save()
                self.stdout.write(self.style.SUCCESS(f"✅ Đã thêm sách: {title} vào Department: {dept_folder}")) 