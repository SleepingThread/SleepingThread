# -*- coding: utf-8 -*-

"""
"""

#import mysql.connector as sql

def escapeText(text):
    text = re.sub(r"\\",r"\\\\",text)
    text = re.sub(r"\"",r'\\"',text)
    return text

def makeMYSQLstring(text):
    text = '"'+escapeText(text)+'"'
    return text
