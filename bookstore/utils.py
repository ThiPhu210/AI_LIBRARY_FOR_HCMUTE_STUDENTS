import requests
import logging
from urllib.parse import quote
from bs4 import BeautifulSoup
import re

# Cấu hình logging chi tiết hơn
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def fetch_book_info_from_isbn(isbn):
    logger.info(f"=== Bắt đầu tìm thông tin sách với ISBN: {isbn} ===")
    
    # Gọi Google Books API
    logger.info("Đang gọi Google Books API...")
    google_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
    try:
        response = requests.get(google_url)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Google Books API response status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi khi gọi Google Books API: {str(e)}")
        return None

    if data.get("totalItems", 0) > 0:
        book_info = data["items"][0]["volumeInfo"]
        logger.info(f"Đã tìm thấy thông tin sách từ Google Books: {book_info.get('title', '')}")
        
        cover_url = None
        
        # Tìm ảnh bìa từ Vinabook
        logger.info("=== Bắt đầu tìm ảnh bìa từ Vinabook ===")
        try:
            vinabook_url = f"https://www.vinabook.com/search?type=product&q={isbn}"
            logger.info(f"Đang gọi Vinabook: {vinabook_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            vinabook_response = requests.get(vinabook_url, headers=headers)
            logger.info(f"Vinabook response status: {vinabook_response.status_code}")
            
            if vinabook_response.status_code == 200:
                soup = BeautifulSoup(vinabook_response.text, 'html.parser')
                img_tag = soup.find('img', {'class': 'product-image'})
                if img_tag and 'src' in img_tag.attrs:
                    cover_url = img_tag['src']
                    logger.info(f"Đã tìm thấy ảnh bìa từ Vinabook: {cover_url}")
                else:
                    logger.warning("Không tìm thấy thẻ img trong trang Vinabook")
            else:
                logger.warning(f"Vinabook trả về status code không hợp lệ: {vinabook_response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi khi gọi Vinabook: {str(e)}")
        except Exception as e:
            logger.error(f"Lỗi không xác định khi xử lý Vinabook: {str(e)}")

        result = {
            "title": book_info.get("title", ""),
            "author": ", ".join(book_info.get("authors", [])),
            "publisher": book_info.get("publisher", ""),
            "year": book_info.get("publishedDate", "")[:4],
            "desc": book_info.get("description", ""),
            "cover_url": cover_url
        }
        
        if not cover_url:
            logger.warning(f"Không thể tìm thấy ảnh bìa cho sách: {result['title']}")
        else:
            logger.info(f"Đã tìm thấy ảnh bìa cho sách: {result['title']}")
            
        logger.info("=== Kết thúc quá trình tìm kiếm ===")
        return result
    else:
        logger.warning(f"Không tìm thấy thông tin sách với ISBN: {isbn}")
        return None
