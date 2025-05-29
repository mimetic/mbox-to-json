# main.py
import os
import errno
import pathlib
import mailbox
import argparse
import re
import pandas as pd
import html2text
from email.header import decode_header
from alive_progress import alive_bar


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

def write_to_disk(part, file_path):
    with open(file_path, 'wb') as f:
        f.write(part.get_payload(decode=True))


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

def extract_attachments_for_message(msg, output_folder, mid):
    """
    Walks a single email.message.Message, saves each attachment
    prefixed with its cleaned Message-ID, and returns a list of
    the filenames actually written.
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
        write_to_disk(part, dest)
        files.append(os.path.basename(dest))

    return files
# ——————————————————————————————————————————————————————————————————————


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i','--input',  default='all.mbox')
    p.add_argument('-o','--output-json', default='out.json')
    p.add_argument('--attachments-dir', default='attachments/')
    return p.parse_args()


def main():
    args = parse_args()

    # prepare output folder for attachments
    os.makedirs(args.attachments_dir, exist_ok=True)

    mbox = mailbox.mbox(args.input)
    mbox_dict = {}

    print("Processing messages and extracting attachments…")
    with alive_bar(len(mbox)) as bar:
        for idx, msg in enumerate(mbox):
            # --- your existing code to turn headers/body into a dict record ---
            record = {
                'From': msg.get('From'),
                'To':   msg.get('To'),
                'Subject': msg.get('Subject'),
                'Date': msg.get('Date'),
                # etc…
            }
            
            # Extract email body and convert to markdown if HTML
            body_content = extract_body(msg)
            if body_content:
                record['Body'] = body_content

            # --- now extract attachments for this message ---
            attached_files = extract_attachments_for_message(
                msg,
                args.attachments_dir,
                idx
            )
            if attached_files:
                record['Attachments'] = attached_files

            mbox_dict[idx] = record
            bar()

    # finally turn your dict into DataFrame and write out
    df = pd.DataFrame.from_dict(mbox_dict, orient='index')
    df.to_json(args.output_json, orient='records', force_ascii=False)
    print(f"Done! JSON written to {args.output_json}")
    print(f"Attachments in   {args.attachments_dir}")


if __name__ == '__main__':
    main()
