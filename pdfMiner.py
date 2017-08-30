# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:23:58 2017


"""

from collections import OrderedDict

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams,LTTextBox,LTTextLine,LTTextLineHorizontal,LTTextBoxHorizontal
from pdfminer.converter import TextConverter
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfpage import PDFPage
from io import StringIO

def to_text(path):
    """Wrapper around pdfminer. Returns whole text as first value, pdf
    layouts with corresponding pages as second"""
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    laparams.all_texts = False
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    deviceLayout=PDFPageAggregator(rsrcmgr,laparams=laparams)
    interpreterLayout= PDFPageInterpreter(rsrcmgr, deviceLayout)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    pages = PDFPage.get_pages(
        fp, pagenos, maxpages=maxpages, password=password,
        caching=caching, check_extractable=True)
    objects = []
    for page_n, page in enumerate(pages):
        interpreter.process_page(page)
        interpreterLayout.process_page(page)
        layout = deviceLayout.get_result()
        objects.append((content_from_layout(layout),page_n))


    fp.close()
    device.close()
    deviceLayout.close()
    string = retstr.getvalue()
    retstr.close()
    return string, objects

def sort_layout(pages):
    """Enables sorting of pdf document layouts based on their relative positions"""

    y0_list = set()
    for page in pages:
        for layout in page[0]:
            y0_list.add(round(1000 -layout.y0+1000*page[1],0))
    dic = OrderedDict()
    for page in pages:
        for layout in page[0]:
            dic.setdefault(round(1000-layout.y0+1000*page[1],0),[]).append((layout.get_text(),layout.x0))

    dicTemp = OrderedDict()
    for key,value in dic.items():
            dicTemp[key] = list(set(value))
    dic = dicTemp
    return dic


def content_from_layout(layout):
    """Get rid of non- textual fields"""
    tcols=[]
    objstack=list(reversed(layout._objs))

    tcols=[]
    while objstack:
        b=objstack.pop()
        if type(b) in [LTTextLine,LTTextLineHorizontal,LTTextBoxHorizontal,LTTextBox]:
            objstack.extend(reversed(b._objs))
            tcols.append(b)
    out = []
    for col in tcols:
        if type(col) in [LTTextLine,LTTextLineHorizontal]:
            out.append(col)
    return out
