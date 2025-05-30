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
__version__ = '1.2.0'

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
    """
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.unicode_snob = True

    body_text = None
    body_html = None

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
                    body_html = decoded_html

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
                body_html = decoded

    # Convert HTML to markdown if necessary
    if body_html:
        raw_text = h.handle(body_html)
    elif body_text:
        raw_text = body_text
    else:
        raw_text = ""

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

    return cleaned_text


def extract_attachments_for_message(msg, output_folder, mid, compress_images=True):
    files = []
    existing = set()

    raw_id = msg.get('Message-ID', '').strip()
    clean_id = raw_id.lstrip('<').rstrip('>') or str(mid)
    counter = 0

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
        compress_images=True
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
        args.attachments_dir = os.path.join(input_dir, 'attachments')

    os.makedirs(args.attachments_dir, exist_ok=True)

    mbox = mailbox.mbox(args.input)
    mbox_dict = {}

    print("Processing messages and extracting attachments…")
    with alive_bar(len(mbox)) as bar:
        for idx, msg in enumerate(mbox):
            from_header = msg.get('From', '')
            display_name, email_addr = email.utils.parseaddr(from_header)
            display_name = ' '.join(display_name.split())

            body_content = extract_body(msg, args.remove_quotes, args.remove_signature)

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
                compress_images=args.compress_images
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
    print()

    df = pd.DataFrame.from_dict(mbox_dict, orient='index')
    df.to_json(args.output_json, orient='records', force_ascii=False)
    print(f"Done! JSON written to {args.output_json}")
    print(f"Attachments in   {args.attachments_dir}")


if __name__ == '__main__':
    main()
