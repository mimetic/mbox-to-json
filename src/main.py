#!/usr/bin/env python3
# main.py
import urllib.parse     # ← for --simplify-links
import os
import errno
import pathlib
import mailbox
import argparse
import re
import pandas as pd
import html2text
import email.utils
import sys

# Package version
__version__ = '1.3'

from email.header import decode_header
from alive_progress import alive_bar
import mimetypes

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pdf2image
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


# ——————————————————————————————————————————————————————————————————————
# 1) Helper functions (from extract.py)

import base64
import re
import uuid

def get_extension(name):
    ext = pathlib.Path(name).suffix
    return ext if len(ext) <= 20 else ''

def filter_fn_characters(s):
    # strip weird whitespace and forbidden chars
    result = re.sub(r'[\t\r\n\v\f]+', ' ', s)
    result = re.sub(r'[/\\\?%\*:\|"<>\0]', '_', result)
    return result

def decode_filename(part, fallback, mid):
    name = part.get_filename()
    if not name:
        return fallback
    dh = decode_header(name)[0]
    text, enc = dh if len(dh) == 2 else (dh[0], None)
    if isinstance(text, bytes):
        try:
            return text.decode(enc or 'utf-8', errors='ignore')
        except Exception:
            return fallback
    return text

def resolve_name_conflicts(folder, name, existing_paths, att_num):
    path = os.path.join(folder, name)
    i = 1
    while os.path.normcase(path) in existing_paths:
        ext = get_extension(name)
        base = os.path.splitext(name)[0]
        suffix = f" ({i})"
        path = os.path.join(folder, base + suffix + ext)
        i += 1
    existing_paths.add(os.path.normcase(path))
    return path

def get_mime_type(content_type, filename):
    """Get standardized MIME type from content type and filename"""
    mime_type = content_type.lower().split(';')[0].strip()
    if not mime_type or mime_type == 'application/octet-stream':
        guessed_type, _ = mimetypes.guess_type(filename)
        if guessed_type:
            mime_type = guessed_type
    return mime_type

def should_compress_file(file_path, mime_type):
    """Check if file should be compressed based on size and type"""
    if not os.path.exists(file_path):
        return False
    file_size = os.path.getsize(file_path)
    return (file_size > 1024 * 1024 and  # 1 MB
            (mime_type.startswith('image/') or mime_type == 'application/pdf'))

def convert_pdf_to_image(pdf_path, output_path):
    """Convert first page of PDF to image using pdf2image"""
    if not HAS_PDF2IMAGE:
        print(f"Warning: pdf2image not available. Cannot convert PDF: {pdf_path}")
        return False
    try:
        images = pdf2image.convert_from_path(pdf_path, first_page=1, last_page=1)
        if images:
            images[0].save(output_path, 'JPEG')
            return True
        return False
    except Exception as e:
        print(f"Failed to convert PDF {pdf_path}: {str(e)}")
        return False

def compress_image(input_path, output_path, max_size=2048, quality=60):
    """Compress image using PIL while maintaining aspect ratio"""
    if not HAS_PIL:
        print(f"Warning: PIL not available. Cannot compress image: {input_path}")
        return False
    try:
        with Image.open(input_path) as img:
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            ratio = min(max_size / max(img.size), 1.0)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
        return True
    except Exception as e:
        print(f"Failed to compress image {input_path}: {str(e)}")
        return False

def write_to_disk(part, file_path, compress=False):
    """Write attachment to disk with optional compression"""
    content = part.get_payload(decode=True)
    if content is None:
        return None

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as f:
        f.write(content)

    if not compress:
        return file_path

    mime_type = get_mime_type(part.get_content_type(), file_path)

    if should_compress_file(file_path, mime_type):
        base_path, ext = os.path.splitext(file_path)
        compressed_path = f"{base_path}_compressed.jpg"

        if mime_type == 'application/pdf' and HAS_PDF2IMAGE:
            if convert_pdf_to_image(file_path, compressed_path):
                if os.path.getsize(compressed_path) > 1024 * 1024:
                    temp_path = f"{base_path}_temp.jpg"
                    os.rename(compressed_path, temp_path)
                    if compress_image(temp_path, compressed_path):
                        os.remove(temp_path)
                        os.remove(file_path)
                        print(f"Compressed PDF: {os.path.basename(file_path)} → {os.path.basename(compressed_path)}")
                        return compressed_path
                    else:
                        os.rename(temp_path, compressed_path)
                        os.remove(file_path)
                        return compressed_path
                else:
                    os.remove(file_path)
                    print(f"Converted PDF: {os.path.basename(file_path)} → {os.path.basename(compressed_path)}")
                    return compressed_path

        elif mime_type.startswith('image/') and HAS_PIL:
            if compress_image(file_path, compressed_path):
                if os.path.getsize(compressed_path) < os.path.getsize(file_path):
                    os.remove(file_path)
                    print(f"Compressed image: {os.path.basename(file_path)} → {os.path.basename(compressed_path)}")
                    return compressed_path
                else:
                    os.remove(compressed_path)

    return file_path


def is_llm_readable(filename, mime_type):
    """
    Determine if a file is likely to be readable by an LLM based on its extension or MIME type.
    
    Args:
        filename: The name of the file
        mime_type: The MIME type of the file
        
    Returns:
        True if the file is likely to be readable by an LLM, False otherwise
    """
    # Get file extension
    ext = os.path.splitext(filename.lower())[1].lstrip('.')
    if not ext and mime_type:
        # Try to guess extension from MIME type
        ext = mimetypes.guess_extension(mime_type.lower())
        if ext:
            ext = ext.lstrip('.')
    
    # Common LLM-readable text formats
    text_formats = {
        # Plain text formats
        'txt', 'text', 'md', 'markdown', 'csv', 'json', 'xml', 'html', 'htm',
        # Code formats
        'py', 'js', 'java', 'c', 'cpp', 'h', 'cs', 'php', 'rb', 'go', 'swift',
        'ts', 'jsx', 'css', 'scss', 'sh', 'bat', 'ps1', 'sql', 'yaml', 'yml',
        # Document formats
        'pdf', 'doc', 'docx', 'rtf', 'odt', 'tex',
        # Presentation formats
        'ppt', 'pptx', 'odp',
        # Spreadsheet formats
        'xls', 'xlsx', 'ods',
        # Image formats
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'svg',
    }
    
    # Check extension
    if ext in text_formats:
        return True
    
    # Check MIME type
    if mime_type:
        mime_type = mime_type.lower()
        if (mime_type.startswith('text/') or
            mime_type.startswith('image/') or
            mime_type == 'application/pdf' or
            'document' in mime_type or
            'spreadsheet' in mime_type or
            'presentation' in mime_type or
            mime_type in [
                'application/json',
                'application/xml',
                'application/javascript',
                'application/markdown',
                'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/vnd.ms-powerpoint',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            ]):
            return True
    
    # Explicitly exclude certain formats
    excluded_formats = {'zip', 'rar', 'tar', 'gz', 'bz2', '7z', 'exe', 'bin', 'iso', 'dmg'}
    if ext in excluded_formats:
        return False
    
    # If the MIME type explicitly indicates a binary or archive format, exclude it
    if mime_type and (
        'zip' in mime_type or
        'compressed' in mime_type or
        'octet-stream' in mime_type or
        'executable' in mime_type or
        'archive' in mime_type
    ):
        return False
    
    # Default to allowing the file if we can't determine
    return True


def extract_inline_base64_images(content, is_html=False):
    """
    Extract base64-encoded images from text content and return cleaned content
    and a list of extracted images.
    
    Args:
        content: The text content to process
        is_html: Whether the content is HTML (True) or plain text (False)
    
    Returns:
        A tuple of (cleaned_content, extracted_images)
        where extracted_images is a list of dicts with 'data', 'mime_type', and 'extension'
    """
    extracted_images = []
    
    if not content:
        return content, extracted_images
    
    if is_html:
        # Match HTML img tags with base64-encoded src
        # This handles both inline and multiline base64 data in HTML
        pattern = re.compile(r'<img[^>]*src=[\'"](data:image/([^;]+);base64,([^"\']+))[\'"][^>]*>', re.DOTALL)
        
        def replace_match(match):
            full_data_uri = match.group(1)
            mime_subtype = match.group(2)
            base64_data = match.group(3)
            
            # Clean up the base64 data (remove whitespace, newlines, etc.)
            cleaned_data = re.sub(r'\s+', '', base64_data)
            
            try:
                # Decode the base64 data to check if it's valid
                decoded_data = base64.b64decode(cleaned_data)
                
                # Only process if we have actual image data
                if decoded_data:
                    # Determine the appropriate extension
                    ext = mime_subtype.lower()
                    if ext == 'jpeg':
                        ext = 'jpg'
                    elif ext == 'svg+xml':
                        ext = 'svg'
                    
                    extracted_images.append({
                        'data': decoded_data,
                        'mime_type': f'image/{mime_subtype}',
                        'extension': ext
                    })
                    
                    # Return an empty img tag or placeholder to indicate where the image was
                    return f'[INLINE_IMAGE_{len(extracted_images)}]'
            except Exception as e:
                print(f"Error decoding base64 image: {str(e)}")
                
            # Return the original tag if there was an error
            return match.group(0)
        
        # Replace all base64 images in the HTML content
        cleaned_content = pattern.sub(replace_match, content)
    else:
        # For plain text content, look for the common patterns in email clients
        # This handles the case where the email is quoted-printable encoded with base64 images
        pattern = re.compile(r'(src=3D["\'](data:image/([^;]+);base64,([^"\']+))["\'])', re.DOTALL)
        
        def replace_match_text(match):
            full_match = match.group(1)
            full_data_uri = match.group(2)
            mime_subtype = match.group(3)
            base64_data = match.group(4)
            
            # Clean up the base64 data (remove =3D, whitespace, newlines, etc.)
            cleaned_data = re.sub(r'(?:=3D|\s+)', '', base64_data)
            
            try:
                # Decode the base64 data to check if it's valid
                decoded_data = base64.b64decode(cleaned_data)
                
                # Only process if we have actual image data
                if decoded_data:
                    # Determine the appropriate extension
                    ext = mime_subtype.lower()
                    if ext == 'jpeg':
                        ext = 'jpg'
                    elif ext == 'svg+xml':
                        ext = 'svg'
                    
                    extracted_images.append({
                        'data': decoded_data,
                        'mime_type': f'image/{mime_subtype}',
                        'extension': ext
                    })
                    
                    # Return a placeholder for the image
                    return f'[INLINE_IMAGE_{len(extracted_images)}]'
            except Exception as e:
                print(f"Error decoding base64 image in plain text: {str(e)}")
                
            # Return the original content if there was an error
            return full_match
        
        # Replace base64 images in the plain text content
        cleaned_content = pattern.sub(replace_match_text, content)
        
        # Also handle the plain base64 blocks that might appear in the text
        # This handles the case where we have multiline base64 data
        multiline_pattern = re.compile(r'data:image/([^;]+);base64,([^\s"\']+(?:\s*=\s*[^\s"\']+)*)', re.DOTALL)
        
        def replace_multiline_match(match):
            mime_subtype = match.group(1)
            base64_data = match.group(2)
            
            # Clean up the base64 data
            cleaned_data = re.sub(r'\s+|=3D|=', '', base64_data)
            
            try:
                # Decode the base64 data to check if it's valid
                decoded_data = base64.b64decode(cleaned_data)
                
                # Only process if we have actual image data
                if decoded_data:
                    # Determine the appropriate extension
                    ext = mime_subtype.lower()
                    if ext == 'jpeg':
                        ext = 'jpg'
                    elif ext == 'svg+xml':
                        ext = 'svg'
                    
                    extracted_images.append({
                        'data': decoded_data,
                        'mime_type': f'image/{mime_subtype}',
                        'extension': ext
                    })
                    
                    # Return a placeholder for the image
                    return f'[INLINE_IMAGE_{len(extracted_images)}]'
            except Exception as e:
                print(f"Error decoding multiline base64 image: {str(e)}")
                
            # Return the original content if there was an error
            return match.group(0)
        
        # Replace multiline base64 images in the plain text content
        cleaned_content = multiline_pattern.sub(replace_multiline_match, cleaned_content)
    
    return cleaned_content, extracted_images


# ——————————————————————————————————————————————————————————————————————
# 2) Link-simplification helpers

def _root_of(url: str) -> str:
    """
    Return just scheme + domain with a trailing slash.
    https://example.com/foo?bar → https://example.com/
    """
    parsed = urllib.parse.urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"

_MD_LINK_RE = re.compile(r'\[([^\]]+)]\((https?://[^\s)]+)\)')
# Modified to avoid incorrect matches with '>>>>' patterns
_RAW_URL_RE = re.compile(r'(?<![\](])(?<![>])\bhttps?://(?:\[[\da-fA-F:\.]+\](?::\d+)?|[^\s)\[\]]+)(?::\d+)?(?:/[^\s)]*)?')

def simplify_links(text: str) -> str:
    """
    • Markdown links: [label](https://site/path) → [label](https://site/)
    • Bare URLs:      https://site/path         → https://site/
    """
    def _md_sub(m):
        label, url = m.groups()
        return f"[{label}]({_root_of(url)})"

    text = _MD_LINK_RE.sub(_md_sub, text)
    text = _RAW_URL_RE.sub(lambda m: _root_of(m.group(0)), text)
    return text.strip()


# ---------------------------------------------------------------------
# regex helpers
_BLANK_QUOTED = re.compile(r'^[ \t]*>+[ \t]*\r?\n', re.MULTILINE)
_RUNS_OF_BLANKS = re.compile(r'(?:\r?\n){2,}')  # 2 or more → 1

def normalize(text: str) -> str:
    text = _BLANK_QUOTED.sub('', text)          # drop >>>>>
    text = _RUNS_OF_BLANKS.sub('\n', text)    # squash blank runs
    return text


# ——————————————————————————————————————————————————————————————————————
# 3) Body and attachment extraction

def clean_quote_markers(text, remove_all_quotes=True):
    """
    Clean email quote markers from text.
    
    Args:
        text: The text to clean
        remove_all_quotes: If True, remove all lines starting with '>' (any number of quotes)
                          If False, only remove lines with '>>' or more quote markers
    
    """
    if not text:
        return ""
    
    # Normalize line endings
    text = text.replace('\r\n', '\n')
    
    # Apply quote filtering
    if remove_all_quotes:
        # Remove all quoted lines
        text = re.sub(r'(?m)^\s*>+.*$\n?', '', text)
    else:
        # Only remove deeply nested quotes ('>>' or more)
        text = re.sub(r'(?m)^\s*>{2,}.*$\n?', '', text)
    
    # Clean up and return
    return text.strip()


def remove_signature(text):
    """
    Remove signature from email body text.
    
    Finds the FIRST occurrence of a line containing exactly "David I. Gross"
    (allowing for decorative characters) and removes that line and all text after it.
    
    Args:
        text: The email body text to process
        
    Returns:
        The text with the signature removed, or the original text if no signature is found
    """
    if not text:
        return ""
    
    # Split the text into lines and process line by line
    lines = text.split('\n')
    
    # Find the FIRST occurrence of a signature line
    for i, line in enumerate(lines):
        # Skip empty lines
        stripped_line = line.strip()
        if not stripped_line:
            continue
        
        # First, handle quoted signatures by removing any leading quote markers
        # This allows us to detect signatures in lines like "> **...David I. Gross...**"
        unquoted_line = re.sub(r'^>+\s*', '', stripped_line)
        
        # Quick initial check - skip lines that definitely can't contain the signature
        lower_line = unquoted_line.lower()
        if 'david' not in lower_line or 'gross' not in lower_line:
            continue
        
        # Simple, effective signature detection:
        # 1. Remove ALL decorative characters at once (*, ., -, _, =, etc.)
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', unquoted_line)
        # 2. Normalize whitespace (convert multiple spaces to single space)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # 3. Convert to lowercase for case-insensitive comparison
        cleaned_lower = cleaned.lower()
        
        # 4. Check for ALL common variations of the signature
        if (cleaned_lower == 'david i gross' or 
            cleaned_lower == 'david i. gross' or
            cleaned_lower == 'davidi gross' or  # In case spacing is lost
            re.match(r'^david\s+i\.?\s+gross$', cleaned_lower)):
            
            # Found a signature line - return everything before this line
            # Preserve all quoted lines before the signature
            if i > 0:
                return '\n'.join(lines[:i]).strip()
            else:
                return ""
    
    # No signature found, return the original text
    return text


def extract_body(msg, remove_all_quotes=False, remove_sig=False):
    """
    Extract the email body and convert HTML to markdown if necessary.
    
    Args:
        msg: The email message to extract body from
        remove_all_quotes: If True, remove all quoted text (lines starting with '>') 
                          If False, only remove deeply nested quotes ('>>' or more)
        remove_sig: If True, remove signature line and everything after it
    
    Returns:
        A tuple of (body_text, extracted_images) where extracted_images is a list of dicts
        containing base64-encoded inline images that were extracted from the body
    """
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.unicode_snob = True

    body_text = None
    body_html = None
    extracted_images = []

    # First extract the raw text without removing quotes
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disp = str(part.get('Content-Disposition', '')).lower()

            if 'attachment' in content_disp:
                continue

            if content_type == 'text/plain' and body_text is None:
                payload = part.get_payload(decode=True)
                if payload:
                    # Don't clean quotes yet - we'll handle that after signature removal
                    body_text = payload.decode('utf-8', errors='replace')

            elif content_type == 'text/html' and body_html is None:
                payload = part.get_payload(decode=True)
                if payload:
                    decoded_html = payload.decode('utf-8', errors='replace')
                    # Extract and clean base64-encoded images from the HTML content
                    cleaned_html, html_images = extract_inline_base64_images(decoded_html, is_html=True)
                    body_html = cleaned_html
                    extracted_images.extend(html_images)

            if body_text and body_html:
                break
    else:
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            decoded = payload.decode('utf-8', errors='replace')
            if content_type == 'text/plain':
                # Don't clean quotes yet
                body_text = decoded
            elif content_type == 'text/html':
                # Extract and clean base64-encoded images from the HTML content
                cleaned_html, html_images = extract_inline_base64_images(decoded, is_html=True)
                body_html = cleaned_html
                extracted_images.extend(html_images)

    # Convert HTML to markdown if necessary
    if body_html:
        raw_text = h.handle(body_html)
    elif body_text:
        raw_text = body_text
    else:
        raw_text = ""
        
    # Check for base64 images in the plain text (may be quoted-printable encoded)
    if raw_text:
        cleaned_text, text_images = extract_inline_base64_images(raw_text, is_html=False)
        raw_text = cleaned_text
        extracted_images.extend(text_images)

    # Now handle signature removal first
    if remove_sig and raw_text:
        # Remove signature line and everything after it
        # This preserves ALL quoted lines before the signature
        cleaned_text = remove_signature(raw_text)
        
        # Don't apply quote cleaning to pre-signature text - keep all quoted lines intact
        # Only apply quote cleaning if we're not removing signatures
    else:
        # No signature removal, apply quote cleaning according to preference
        cleaned_text = clean_quote_markers(raw_text, remove_all_quotes)
    
    # Remove consecutive empty lines (also with quoted with >)
    cleaned_text = normalize(cleaned_text)

    return cleaned_text, extracted_images


def save_inline_base64_image(image_data, output_folder, clean_id, counter, existing):
    """
    Save a base64-encoded inline image to disk.
    
    Args:
        image_data: Dict containing 'data', 'mime_type', and 'extension'
        output_folder: Directory to save the image
        clean_id: Message ID for filename prefix
        counter: Counter for attachment numbering
        existing: Set of existing filenames to avoid duplicates
        
    Returns:
        The filename of the saved image
    """
    # Generate a unique filename
    filename = f"inline_image_{counter}.{image_data['extension']}"
    prefixed = f"{clean_id} {filename}"
    
    # Resolve name conflicts
    dest = resolve_name_conflicts(output_folder, prefixed, existing, counter)
    
    # Write the image data to disk
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as f:
        f.write(image_data['data'])
    
    # Apply compression if it's an image and PIL is available
    if HAS_PIL and image_data['mime_type'].startswith('image/'):
        mime_type = image_data['mime_type']
        if should_compress_file(dest, mime_type):
            base_path, ext = os.path.splitext(dest)
            compressed_path = f"{base_path}_compressed.jpg"
            
            if compress_image(dest, compressed_path):
                if os.path.getsize(compressed_path) < os.path.getsize(dest):
                    os.remove(dest)
                    print(f"Compressed inline image: {os.path.basename(dest)} → {os.path.basename(compressed_path)}")
                    return os.path.basename(compressed_path)
                else:
                    os.remove(compressed_path)
    
    return os.path.basename(dest)


def extract_attachments_for_message(msg, output_folder, mid, compress_images=True, inline_images=None, only_llm_attachments=True):
    """
    Extract attachments and inline images from an email message.
    
    Args:
        msg: The email message
        output_folder: Directory to save attachments
        mid: Message index
        compress_images: Whether to compress large images
        inline_images: List of inline base64-encoded images to save
        only_llm_attachments: If True, only save attachments that are LLM-readable
        
    Returns:
        List of attachment filenames
    """
    files = []
    existing = set()

    raw_id = msg.get('Message-ID', '').strip()
    clean_id = raw_id.lstrip('<').rstrip('>') or str(mid)
    counter = 0
    
    # First save any inline base64-encoded images
    if inline_images:
        for image_data in inline_images:
            # Images are generally LLM-readable, but we'll check anyway if the filter is enabled
            if not only_llm_attachments or is_llm_readable(f"image.{image_data['extension']}", image_data['mime_type']):
                counter += 1
                filename = save_inline_base64_image(image_data, output_folder, clean_id, counter, existing)
                files.append(filename)

    for part in msg.walk():
        if part.is_multipart():
            continue

        disp = part.get_content_disposition()
        name = part.get_filename()
        is_attachment = (
            disp == 'attachment'
            or (name and disp != 'inline')
            or part.get_content_type().startswith(('application/', 'audio/', 'video/', 'model/'))
        )
        if not is_attachment:
            continue

        if name and name.lower().endswith('.asc'):
            continue

        content = part.get_payload(decode=True)
        if content is None:
            continue

        if part.get_content_type().startswith('image/') and len(content) < 1024:
            continue

        # Check if the attachment is LLM-readable if filtering is enabled
        mime_type = part.get_content_type()
        if only_llm_attachments and not is_llm_readable(name or "", mime_type):
            print(f"Skipping non-LLM-readable attachment: {name or 'unnamed'} ({mime_type})")
            continue

        counter += 1
        num_str = f"{counter}"
        fname = decode_filename(part, num_str, mid)
        fname = filter_fn_characters(fname)
        prefixed = f"{clean_id} {fname}"

        dest = resolve_name_conflicts(output_folder, prefixed, existing, counter)
        final_path = write_to_disk(part, dest, compress=compress_images)
        if final_path:
            files.append(os.path.basename(final_path))

    return files


# ——————————————————————————————————————————————————————————————————————
# 4) CLI parsing

def parse_args():
    p = argparse.ArgumentParser()
    
    # Content processing options
    content_group = p.add_argument_group('Content processing options')
    
    # Option pairs for more intuitive enabling/disabling
    link_group = content_group.add_mutually_exclusive_group()
    link_group.add_argument('--simplify-links', dest='simplify_links', action='store_true',
                           help='Trim links to domain only (default: enabled)')
    link_group.add_argument('--no-simplify-links', dest='simplify_links', action='store_false',
                           help='Keep full links intact')
    
    quote_group = content_group.add_mutually_exclusive_group()
    quote_group.add_argument('--remove-quotes', dest='remove_quotes', action='store_true',
                            help='Remove quoted text (default: enabled)')
    quote_group.add_argument('--no-remove-quotes', dest='remove_quotes', action='store_false',
                            help='Keep quoted text in message bodies')
    
    sig_group = content_group.add_mutually_exclusive_group()
    sig_group.add_argument('--remove-signature', dest='remove_signature', action='store_true',
                          help='Remove signature and text after it (default: enabled)')
    sig_group.add_argument('--no-remove-signature', dest='remove_signature', action='store_false',
                          help='Keep signature in message bodies')
    
    img_group = content_group.add_mutually_exclusive_group()
    img_group.add_argument('--compress-images', dest='compress_images', action='store_true',
                          help='Compress large images & PDFs (default: enabled)')
    img_group.add_argument('--no-compress-images', dest='compress_images', action='store_false',
                          help='Keep original image files without compression')
    
    llm_group = content_group.add_mutually_exclusive_group()
    llm_group.add_argument('--only-llm-attachments', dest='only_llm_attachments', action='store_true',
                          help='Only save attachments that an LLM can read (default: enabled)')
    llm_group.add_argument('--no-only-llm-attachments', dest='only_llm_attachments', action='store_false',
                          help='Save all attachments regardless of readability by LLMs')
    
    # File options
    file_group = p.add_argument_group('File options')
    file_group.add_argument('-i', '--input', default='all.mbox',
                           help='Input mbox file (default: all.mbox)')
    file_group.add_argument('-o', '--output-json', default=None,
                           help='Output JSON file (defaults beside input file)')
    file_group.add_argument('-a', '--attachments-dir', default=None,
                           help='Directory for attachments (defaults beside input file)')
    
    # Other options
    p.add_argument('-v', '--version', action='version',
                   version=f'mbox-to-json {__version__}',
                   help='Show program version and exit')
    
    # Set defaults
    p.set_defaults(
        simplify_links=True,
        remove_quotes=True,
        remove_signature=True,
        compress_images=True,
        only_llm_attachments=True
    )
    return p.parse_args()


# ——————————————————————————————————————————————————————————————————————
# 5) Main

def main():
    args = parse_args()

    input_dir = os.path.dirname(args.input) or '.'

    if args.output_json is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output_json = os.path.join(input_dir, f"{base}.json")

    if args.attachments_dir is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.attachments_dir = os.path.join(input_dir, f"{base}_attachments")

    os.makedirs(args.attachments_dir, exist_ok=True)

    mbox = mailbox.mbox(args.input)
    mbox_dict = {}

    print("Processing messages and extracting attachments…")
    with alive_bar(len(mbox)) as bar:
        for idx, msg in enumerate(mbox):
            from_header = msg.get('From', '')
            display_name, email_addr = email.utils.parseaddr(from_header)
            display_name = ' '.join(display_name.split())

            body_content, inline_images = extract_body(msg, args.remove_quotes, args.remove_signature)

            # Skip messages with no usable body
            if not body_content or not body_content.strip():
                bar()
                continue

            # Optional link simplification
            if args.simplify_links:
                body_content = simplify_links(body_content)

            record = {
                'from': from_header,
                'to': msg.get('To'),
                'subject': msg.get('Subject'),
                'date': msg.get('Date'),
                'display-name': display_name,
                'body': body_content
            }

            attached_files = extract_attachments_for_message(
                msg,
                args.attachments_dir,
                idx,
                compress_images=args.compress_images,
                inline_images=inline_images,
                only_llm_attachments=args.only_llm_attachments
            )
            if attached_files:
                record['attachments'] = attached_files

            mbox_dict[idx] = record
            bar()

    # Print summary of options used
    print("\nOptions summary:")
    print(f"  {'Simplify links:':25} {'Enabled' if args.simplify_links else 'Disabled'}")
    print(f"  {'Remove quoted text:':25} {'Enabled' if args.remove_quotes else 'Disabled'}")
    print(f"  {'Remove signature:':25} {'Enabled' if args.remove_signature else 'Disabled'}")
    print(f"  {'Compress images:':25} {'Enabled' if args.compress_images else 'Disabled'}")
    print(f"  {'Only LLM attachments:':25} {'Enabled' if args.only_llm_attachments else 'Disabled'}")
    print()

    df = pd.DataFrame.from_dict(mbox_dict, orient='index')
    df.to_json(args.output_json, orient='records', force_ascii=False)
    print(f"Done! JSON written to {args.output_json}")
    print(f"Attachments in   {args.attachments_dir}")


if __name__ == '__main__':
    main()
