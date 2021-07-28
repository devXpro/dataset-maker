import os
from pdf2image import convert_from_path
import pathlib
root_path = pathlib.Path().resolve()


def get_pdf_files(folder):
    i = 0
    for path, subdirs, files in os.walk(folder):
        for name in files:
            if name.split('.')[1] == 'pdf':
                file_path = os.path.join(path, name)
                file_full_path = str(root_path) + '/' + file_path
                pages = convert_from_path(file_full_path)
                for page in pages:
                    page.save('out/' + str(i) + '.jpg', 'JPEG')
                    i = i + 1


get_pdf_files('pdf')
