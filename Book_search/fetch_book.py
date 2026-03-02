import requests 
import logging 
import yaml 
import json
import re
import datetime
from tqdm import tqdm
from pathlib import Path
from urllib.parse import quote

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

config_dir = Path(__file__).parent
config_file = Path(__file__).parent.parent / "setup.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)
    BASE_PATH = config["BASE_PATH"]
    google_book_api_key = config["Google_book_api_key"]
    downloaded_rawbook_dir = config["Downloaded_rawbook_dir"]
    downloaded_data_meta = config["Downloaded_data_meta"]
    book_data = config["Book_data"]
    archive_url = config["Archive_url"]
    Download_formats = config["Download_formats"]
    archive_metadata_url = config["Archive_metadata_url"]

def get_user_input():
    user_input = input("Enter a book title:")
    return user_input 

def receive_book(google_book_api_key, user_input):
    encoded_input = quote(user_input)
    url = f"https://www.googleapis.com/books/v1/volumes?q=title:{encoded_input}&key={google_book_api_key}&maxResults=20"
    
    try:
        response = requests.get(url, timeout=10)
        logging.info(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                if "error" in data:
                    logging.error(f"API Error: {data.get('error', {}).get('message', 'Unknown error')}")
                    return None
                logging.info("API key is valid")
                return data
            except json.JSONDecodeError:
                logging.error("Invalid JSON response from API")
                return None
        elif response.status_code == 401 or response.status_code == 403:
            logging.error("API key is invalid or unauthorized")
            return None
        else:
            logging.error(f"API request failed with status code: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error: {str(e)}")
        return None

def extract_books_info(search_results):
    if not search_results or 'items' not in search_results:
        return []

    book_fields = book_data
    books = []
    for item in search_results['items']:
        volume_info = item.get('volumeInfo', {})
        access_info = item.get('accessInfo', {})

        current_book = {key: volume_info.get(key, None) for key in book_fields}
        current_book['id'] = item.get('id')
        current_book['selfLink'] = item.get('selfLink')
        current_book['Links_to_access'] = {}
        current_book['accessInfo'] = access_info

        current_book['webReaderLink'] = access_info.get('webReaderLink')
        current_book['viewability'] = access_info.get('viewability')
        current_book['publicDomain'] = access_info.get('publicDomain')
        current_book['previewLink'] = volume_info.get('previewLink')
        current_book['infoLink'] = volume_info.get('infoLink')

        identifiers = volume_info.get('industryIdentifiers', [])
        for identifier in identifiers:
            if identifier.get('type') == 'ISBN_13':
                current_book['isbn'] = identifier.get('identifier')
                break
            elif identifier.get('type') == 'ISBN_10' and not current_book.get('isbn'):
                current_book['isbn'] = identifier.get('identifier')
                break

        for format in ['pdf', 'epub']:
            download_link = access_info.get(format, {}).get('downloadLink')
            if download_link:
                current_book['Links_to_access'][format] = download_link
                    
        books.append(current_book)

    return books

def check_and_display_viewability(current_book):
    access_info = current_book.get('accessInfo', {})

    viewability_info = {
    'viewability': access_info.get('viewability', 'No_Pages'),
    'is_public_domain': access_info.get('publicDomain', False),
    'pdf_available': access_info.get('pdf', {}).get('isAvailable', False),
    'epub_available': access_info.get('epub', {}).get('isAvailable', False)
    }

    print(f"Viewability: {viewability_info['viewability']}")
    print(f"Public Domain: {viewability_info['is_public_domain']}")
    print(f"PDF Available: {viewability_info['pdf_available']}")
    print(f"EPUB Available: {viewability_info['epub_available']}")

    current_book.update(viewability_info)

    return viewability_info

def display_book_info(i, current_book):
    print(f"\n{'='*60}")
    print(f"Book {i}: {current_book.get('title', 'Unknown')}")
    print(f"Authors: {current_book.get('authors', 'Unknown')}")
    print(f"Summary: {current_book.get('description', 'No summary available')}")
    print(f"{'='*60}")

def create_book_summary(current_book, viewability_info):
    summary = {
        'title': current_book.get('title', 'Unknown'),
        'authors': current_book.get('authors', 'Unknown'),
        'description': current_book.get('description', 'No summary available'),
        'viewability': viewability_info['viewability'],
        'is_public_domain': viewability_info['is_public_domain'],
        'pdf_available': viewability_info['pdf_available'],
        'epub_available': viewability_info['epub_available'],
        'Links_to_access': current_book.get('Links_to_access', {})
    }
    
    if current_book.get('webReaderLink'):
        summary['Links_to_access']['read_online'] = current_book.get('webReaderLink')
    if current_book.get('previewLink'):
        summary['Links_to_access']['preview'] = current_book.get('previewLink')
    if current_book.get('infoLink'):
        summary['Links_to_access']['info_page'] = current_book.get('infoLink')
    
    return summary

def process_book(current_book, index):
    display_book_info(index, current_book)

    viewability_info = check_and_display_viewability(current_book)

    if viewability_info['is_public_domain']:
        print("\nPublic domain - Access Links:")
        if current_book.get('webReaderLink'):
            print (f" Read Online: {current_book.get('webReaderLink')}")
        if current_book.get('previewLink'):
            print (f" Preview: {current_book.get('previewLink')}")
        if current_book.get('infoLink'):
            print (f" Info: {current_book.get('infoLink')}")

    if current_book.get('Links_to_access'):
        print(f"\nDownload links found: {current_book['Links_to_access']}")
    else:
        print("\nChecking Archive.org for the book")
    
    print()

    return create_book_summary(current_book, viewability_info)

def save_google_books(books_to_save, user_input,output_dir):
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    public_domain_books = [book for book in books_to_save if book.get('is_public_domain')]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    google_books_file = save_path / f"google_books_{timestamp}.json"

    with open(google_books_file, "w") as file:
        json.dump({
        'search_query': user_input,
        'total_books': len(public_domain_books),
        'timestamp': timestamp,
        'books': public_domain_books
        }, 
        file, indent=4, ensure_ascii=False)

    print("\n" + "="*60)
    print(f"Saved {len(public_domain_books)} books to {google_books_file}")
    print("\n" + "="*60)

    return google_books_file

def search_in_archive(book_info, archive_url):

    title = book_info.get('title', 'Unknown')
    authors = book_info.get('authors', '[]')
    url = archive_url

    author_str = None
    if isinstance(authors, list) and len(authors) > 0:
        author_str = authors[0]
    elif isinstance(authors, str):
        author_str = authors

    search_query = f'title:"{title}" AND mediatype:texts' 
        
    params = {
        "q": search_query,
        "fl": "identifier,title,creator,downloads",
        "sort": "downloads desc",
        "rows": 10,
        "output": "json"
    }

    try: 
        logging.info(f"Searching in Archive.org for: {title} only")
        response = requests.get(archive_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        docs = data.get('response', {}).get('docs', [])
        num_found = data.get('response', {}).get('numFound', 0)

        if docs:
            logging.info(f"Found {len(docs)} results with title only")
            return docs 

        if author_str:
            logging.info(f"No results found with title only. Searching with author: {author_str}")
            search_query = f'title:"{title}" AND creator:"{author_str}" AND mediatype:texts'
            params["q"] = search_query

            response = requests.get(archive_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            docs = data.get('response', {}).get('docs', [])
            num_found = data.get('response', {}).get('numFound', 0)

            if docs:
                logging.info(f"Found {num_found} result(s) with title and author")
                return docs
            else:
                logging.warning(f"No results found for '{title}' by '{author_str}' in Archive.org")
                return None
        else:
            logging.warning(f"No results found for '{title}' in Archive.org (no author metadata)")
            return None
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error searching in Archive.org: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error searching in Archive.org: {str(e)}")
        return None

_FORMAT_MAP = {
    "pdf":  ([".pdf"],  ["PDF"]),
    "epub": ([".epub"], ["EPUB"]),
}

def _find_file_for_formats(files, formats):
    for fmt in formats:
        fmt_info = _FORMAT_MAP.get(fmt.lower())
        for file_info in files:
            name = file_info.get('name', '')
            ftype = file_info.get('format', '')
            if fmt_info:
                exts, types = fmt_info
                if any(name.lower().endswith(e) for e in exts) or ftype in types:
                    return name
            elif name.lower().endswith(f'.{fmt.lower()}'):
                return name
    return None

def download_archive_book(docs, output_dir, formats=None):

    if formats is None:
        formats = Download_formats

    if not docs or len(docs) == 0:
        logging.error("No archive search results found")
        return None 

    archive_book = docs[0]
    identifier = archive_book.get('identifier')
    title = archive_book.get('title', 'Unknown')

    if not identifier:
        logging.error("No identifier found for the archive book")
        return None 

    logging.info(f"Downloading from Archive.org: {identifier} - {title}")

    try:
        metadata_url = f"{archive_metadata_url}{identifier}"
        metadata_response = requests.get(metadata_url, timeout=10)
        metadata_response.raise_for_status()

        metadata_data = metadata_response.json()
        files = metadata_data.get('files', [])

        matched_file = _find_file_for_formats(files, formats)

        if not matched_file:
            logging.warning(f"No file in formats {formats} for identifier: {identifier}")
            return None
            
        logging.info(f"Found file: {matched_file}")

        download_url = f"https://archive.org/download/{identifier}/{matched_file}"

        logging.info(f"Downloading from Archive.org: {download_url}")
        response = requests.get(download_url, timeout=60, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))  

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
        safe_title = safe_title.replace(' ', '_')
        safe_title = safe_title[:100]

        ext = Path(matched_file).suffix or f".{formats[0]}"
        filename = f"archive__{identifier}__{safe_title}{ext}"
        file_path = output_path / filename

        with open(file_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename, 
            ncols=100, unit_divisor=1024, ascii=False) as pbar:
                for chunk in response.iter_content(chunk_size=10485760):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        file_size = file_path.stat().st_size
        logging.info(f"Saved {filename} {file_size} bytes")
        print(f"Downloaded: {filename} ({file_size} bytes)") 
        
        return file_path
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error downloading from Archive.org: {str(e)}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON response from Archive.org metadata API")
        return None
    except Exception as e:
        logging.error(f"Error downloading from Archive.org: {str(e)}")
        return None

if __name__ == "__main__":
    downloaded_rawbook_dir = Path(BASE_PATH) / downloaded_rawbook_dir
    downloaded_data_meta = Path(BASE_PATH) / downloaded_data_meta

    user_input = get_user_input()
    search_results = receive_book(google_book_api_key, user_input)

    if not search_results:
        print("No books found")
        exit(1)

    books = extract_books_info(search_results)

    if not books:
        print("No books could be extracted")
        exit(1)
        
    print(f"\nFound {len(books)} book(s):\n")

    books_to_save = []
    downloaded_count = 0
    failed_count = 0 

    for i, current_book in enumerate(books, start=1):
        book_summary = process_book(current_book, i)
        
        if book_summary.get('is_public_domain'):
            books_to_save.append(book_summary)
            
            print(f"\n{'='*60}")
            print(f"Processing: {book_summary.get('title', 'Unknown')}")
            print(f"{'='*60}")

            archive_search = search_in_archive(current_book, archive_url)

            if archive_search:
                print(f"Found {len(archive_search)} results in Archive.org")
                downloaded_file = download_archive_book(archive_search, downloaded_rawbook_dir)
                if downloaded_file:
                    print(f"Downloaded Archive Book: {downloaded_file}")
                    book_summary['download_source'] = 'archive.org'
                    book_summary['file_path'] = str(downloaded_file)
                    book_summary['file_name'] = downloaded_file.name
                    downloaded_count += 1
                else:
                    print("Failed to download Archive Book")
                    failed_count += 1
            else:
                print("No results found in Archive.org")
                failed_count += 1
        else:
            print(f"\nSkipping '{book_summary.get('title', 'Unknown')}' - Not public domain")

    if books_to_save:
        google_books_file = save_google_books(books_to_save, user_input, downloaded_data_meta)
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total books found: {len(books)}")
        print(f"Public domain books: {len(books_to_save)}")
        print(f"Successfully downloaded: {downloaded_count}")
        print(f"Failed to download: {failed_count}")
        print(f"Metadata saved to: {google_books_file}")
        print("="*60)
    else:
        print("\nNo public domain books found to process")

