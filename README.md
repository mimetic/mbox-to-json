# MBOX to JSON

A command line tool to convert MBOX file to JSON with attachments.
  
## Example 
```
$ python main.py -i /Volumes/Projects/eMail/code/test_box/mbox -o /Volumes/Projects/eMail/code/test_output/mbox.json
```

## Built With \- [Python](https://www.python.org/)
<<<<<<< HEAD
=======
This script was built on Python.
ChatGPT wrote almost all of it.
>>>>>>> 4cd45ef (Readme updated)

## Install
Download the repository, e.g., to your Git directory. Use pip to install it into your Python environment.
```
pip install /Users/david/Git/mbox-to-json
```  

## Usage \- Help Function 

### Help
Show the options.
```mbox-to-json -h ``` 

### Conversion
To convert an MBOX file to JSON, provide the MBOX file path. Output JSON file would be in
the same location as the input file. 
You can run the command from command line (be sure to activate your python environment, e.g. ``conda activate myenv``)
```
mbox-to-json -i /Users/david/mail/mailbox.mbox
``` 

Other options:
```
mbox-to-json \
   -i /Users/david/mail/mailbox.mbox \
   -o /Users/david/mail/json \
   -a /Users/david/mail/attachments \
   --compress-images
``` 

### Attachments
- Use **`-a`** to specify the path to the attachments output folder.
- By **default** the attachments are put in an "attachments" folder in the same directory as the JSON file.
- Attachment file names are prefixed with the email's ID, which we assume is unique.

## License Distributed under the MIT License. 
See [`LICENSE.txt`](https://github.com/PS1607/mbox-to-json/blob/main/LICENSE.txt) for more information.