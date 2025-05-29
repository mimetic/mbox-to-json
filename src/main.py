# main.py
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
__version__ = '1.1.0'
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
        except:
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
        # Try to guess from filename if content_type is generic
        guessed_type, _ = mimetypes.guess_type(filename)
        if guessed_type:
            mime_type = guessed_type
    return mime_type

def should_compress_file(file_path, mime_type):
    """Check if file should be compressed based on size and type"""
    if not os.path.exists(file_path):
        return False
    file_size = os.path.getsize(file_path)
    return (file_size > 1024 * 1024 and  # 1MB
            (mime_type.startswith('image/') or mime_type == 'application/pdf'))

def convert_pdf_to_image(pdf_path, output_path):
    """Convert first page of PDF to image using pdf2image"""
    if not HAS_PDF2IMAGE:
        print(f"Warning: pdf2image not available. Cannot convert PDF: {pdf_path}")
        return False
    
    try:
        images = pdf2image.convert_from_path(pdf_path, first_page=1, last_page=1)
        if images and len(images) > 0:
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
            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate new dimensions
            ratio = min(max_size/max(img.size[0], img.size[1]), 1.0)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            
            # Resize and save
            img = img.resize(new_size, Image.LANCZOS)
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
        return True
    except Exception as e:
        print(f"Failed to compress image {input_path}: {str(e)}")
        return False

def write_to_disk(part, file_path, compress=False):
    """Write attachment to disk with optional compression"""
    content = part.get_payload(decode=True)
    
    # Write original file first
    with open(file_path, 'wb') as f:
        f.write(content)
    
    if not compress:
        return file_path
    
    # Get MIME type for compression decision
    mime_type = get_mime_type(part.get_content_type(), file_path)
    
    if should_compress_file(file_path, mime_type):
        # Prepare paths for compressed files
        base_path, ext = os.path.splitext(file_path)
        compressed_path = f"{base_path}_compressed.jpg"
        
        # Handle PDFs
        if mime_type == 'application/pdf' and HAS_PDF2IMAGE:
            pdf_success = convert_pdf_to_image(file_path, compressed_path)
            if pdf_success:
                # Further compress the converted image if it's still large
                if os.path.getsize(compressed_path) > 1024 * 1024:
                    temp_path = f"{base_path}_temp.jpg"
                    os.rename(compressed_path, temp_path)
                    if compress_image(temp_path, compressed_path):
                        os.remove(temp_path)
                        os.remove(file_path)  # Remove original PDF if compression successful
                        print(f"Compressed PDF: {os.path.basename(file_path)} → {os.path.basename(compressed_path)}")
                        return compressed_path
                    else:
                        # If second compression failed, use the first converted image
                        os.rename(temp_path, compressed_path)
                        os.remove(file_path)
                        return compressed_path
                else:
                    # PDF was converted and is small enough
                    os.remove(file_path)  # Remove original if conversion successful
                    print(f"Converted PDF: {os.path.basename(file_path)} → {os.path.basename(compressed_path)}")
                    return compressed_path
        
        # Handle regular images
        elif mime_type.startswith('image/') and HAS_PIL:
            if compress_image(file_path, compressed_path):
                # Only keep compressed version if it's actually smaller
                if os.path.getsize(compressed_path) < os.path.getsize(file_path):
                    os.remove(file_path)  # Remove original if compression successful
                    print(f"Compressed image: {os.path.basename(file_path)} → {os.path.basename(compressed_path)}")
                    return compressed_path
                else:
                    # Compressed version isn't smaller, keep original
                    os.remove(compressed_path)
                    
    # Return original path if no compression happened
    return file_path


def extract_body(msg):
    """
    Extract the email body and convert HTML to markdown if necessary.
    Returns the body content as a string.
    """
    # Initialize HTML to Text converter
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.unicode_snob = True
    
    body_text = None
    body_html = None
    
    # Handle multipart messages
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get('Content-Disposition', '')).lower()
            
            # Skip attachments
            if 'attachment' in content_disposition:
                continue
                
            if content_type == 'text/plain' and not body_text:
                payload = part.get_payload(decode=True)
                if payload:
                    body_text = payload.decode('utf-8', errors='replace')
            
            elif content_type == 'text/html' and not body_html:
                payload = part.get_payload(decode=True)
                if payload:
                    body_html = payload.decode('utf-8', errors='replace')
                    
            # If we have both parts, we can stop
            if body_text and body_html:
                break
    else:
        # Not multipart - just get the payload
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        
        if payload:
            decoded_payload = payload.decode('utf-8', errors='replace')
            if content_type == 'text/plain':
                body_text = decoded_payload
            elif content_type == 'text/html':
                body_html = decoded_payload
    
    # Prefer HTML converted to markdown, fall back to plain text
    if body_html:
        return h.handle(body_html)
    elif body_text:
        return body_text
    else:
        return ""

def extract_attachments_for_message(msg, output_folder, mid, compress_images=False):
    """
    Walks a single email.message.Message, saves each attachment
    prefixed with its cleaned Message-ID, and returns a list of
    the filenames actually written.
    
    If compress_images is True, images and PDFs larger than 1MB will be 
    compressed to JPEG format with a maximum dimension of 2048px.
    """
    files = []
    existing = set()
    # pull real Message-ID header & strip < >
    raw_id = msg.get('Message-ID','').strip()
    clean_id = raw_id.lstrip('<').rstrip('>') or str(mid)

    # count attachments so we can fallback to numbering
    counter = 0

    for part in msg.walk():
        if part.is_multipart():
            continue

        disp = part.get_content_disposition()
        name = part.get_filename()
        is_attachment = (
            disp == 'attachment'
            or (name and disp != 'inline')
            or (part.get_content_type().startswith(('application/','audio/','video/','model/')))
        )
        if not is_attachment:
            continue

        counter += 1
        num_str = f"{counter}"
        # decode, sanitize, then prefix
        fname = decode_filename(part, num_str, mid)
        fname = filter_fn_characters(fname)
        prefixed = f"{clean_id} {fname}"

        dest = resolve_name_conflicts(output_folder, prefixed, existing, counter)
        final_path = write_to_disk(part, dest, compress=compress_images)
        files.append(os.path.basename(final_path))

    return files
# ——————————————————————————————————————————————————————————————————————


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i','--input',  default='all.mbox')
    p.add_argument('-o','--output-json', default='out.json')
    p.add_argument('--attachments-dir', default=None, help='Directory for attachments. If not specified, will be created in same directory as output JSON.')
    p.add_argument('--compress-images', action='store_true', help='Compress images and PDFs larger than 1MB to JPEG format (max 2048px, quality 60)')
    p.add_argument('-v', '--version', action='version', version=f'mbox-to-json {__version__}', help='Show program version and exit')
    return p.parse_args()


def main():
    args = parse_args()

    # If attachments directory not specified, put it in same directory as output JSON
    if args.attachments_dir is None:
        # Get the directory of the output JSON file
        output_dir = os.path.dirname(args.output_json)
        if output_dir == '':
            output_dir = '.'  # If no directory specified, use current directory
        
        # Create 'attachments' folder in that directory
        args.attachments_dir = os.path.join(output_dir, 'attachments')
    
    # prepare output folder for attachments
    os.makedirs(args.attachments_dir, exist_ok=True)

    mbox = mailbox.mbox(args.input)
    mbox_dict = {}

    print("Processing messages and extracting attachments…")
    with alive_bar(len(mbox)) as bar:
        for idx, msg in enumerate(mbox):
            # --- your existing code to turn headers/body into a dict record ---
            from_header = msg.get('From', '')
            display_name, email_addr = email.utils.parseaddr(from_header)
            
            # Normalize display name (remove extra spaces, etc.)
            display_name = ' '.join(display_name.split())
            
            record = {
                'from': from_header,
                'to': msg.get('To'),
                'subject': msg.get('Subject'),
                'date': msg.get('Date'),
                'display-name': display_name,
                # etc…
            }
            
            # Extract email body and convert to markdown if HTML
            body_content = extract_body(msg)
            if body_content:
                record['body'] = body_content

            # --- now extract attachments for this message ---
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

    # finally turn your dict into DataFrame and write out
    df = pd.DataFrame.from_dict(mbox_dict, orient='index')
    df.to_json(args.output_json, orient='records', force_ascii=False)
    print(f"Done! JSON written to {args.output_json}")
    print(f"Attachments in   {args.attachments_dir}")


if __name__ == '__main__':
    main()
