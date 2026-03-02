import os

# Extensions for plain text files
TEXT_EXTENSIONS = {
    '.txt', '.md', '.markdown', '.rtf', '.tex', '.log',
    '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rb', '.php', '.sh', '.bat', '.ps1',
    '.rs', '.swift', '.kt', '.html', '.htm', '.css', '.scss', '.sass', '.less', '.svg',
    '.sql'
}

# Extensions for structured data files (to be parsed with specialized libraries)
STRUCTURED_EXTENSIONS = {
    '.csv', '.json', '.xml', '.yaml', '.yml', '.toml',
    '.ini', '.cfg', '.conf'
}

# Extensions for binary files
BINARY_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.ico',
    '.mp3', '.wav', '.flac', '.ogg', '.mp4', '.avi', '.mkv', '.mov', '.webm',
    '.zip', '.gz', '.tar', '.rar', '.7z', '.bz2',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.db', '.sqlite', '.sqlite3',
    '.exe', '.dll', '.so', '.a', '.lib', '.jar',
    '.ttf', '.otf', '.woff', '.woff2', '.bin', '.dat', '.iso'
}

def classify_file_by_extension(file_path: str) -> str:
    """
    Classify a file based on its extension.

    Returns one of:
      - 'text'       : plain text files
      - 'structured' : structured data files (CSV, JSON, XML, etc.)
      - 'binary'     : binary files (images, archives, executables, etc.)
      - 'unknown'    : files with no recognized extension
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if not ext:
        return 'unknown'
    if ext in BINARY_EXTENSIONS:
        return 'binary'
    if ext in STRUCTURED_EXTENSIONS:
        return 'structured'
    if ext in TEXT_EXTENSIONS:
        return 'text'
    return 'unknown'

# Example usage
if __name__ == '__main__':
    sample_files = [
        'document.txt',
        'spreadsheet.csv',
        'photo.jpg',
        'archive.tar.gz',
        'script.py',
        'settings.yaml',
        'LICENSE'
    ]
    for filename in sample_files:
        category = classify_file_by_extension(filename)
        print(f"{filename} -> {category}")