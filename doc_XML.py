# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:12:33 2017


"""

import docx
from docx import Document
from docx.oxml.shared import qn

def docx_to_text(doc):
    """Parses docx's underlying xml to text"""
    doc_element = doc.element
    all_list = []
    for paragraph in doc_element.body:
        par_list = paragraph.findall('.//' + qn('w:t'))
        par_list = list(map(lambda x: x.text, par_list))
        all_list.append(par_list)
    all_list = list(map(lambda x: ' '.join(x),all_list))
    text = '\n'.join(all_list)
    return text


if __name__ == '__main__':
    doc = Document(r'path')
    text =docx_to_text(doc)

