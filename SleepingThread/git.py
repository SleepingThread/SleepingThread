# -*- coding: utf-8 -*-

import subprocess

def gitstatus():
    """
    return:
        if no git:
            None
        git exists:
            nochanges - all commited
            untracked - there are untracked files
            changes - there are uncommited changes
    """
    process = subprocess.Popen("git status | tail -n 1",shell=True,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    out, err = process.communicate()

    if err=='':
        str1 = "nothing to commit"
        str2 = "nothing added to commit"
        str3 = "no changes added to commit"
        if out[:len(str1)]==str1:
            #"All commited"
            return 'nochanges'
        elif out[:len(str2)]==str2 or out[:len(str3)]==str3:
            #"Something untracked"
            return 'untracked'
        else:
            #"There are changes"
            return 'changes'

    else:
        #"NO GIT REPO"
        return 'nogit'

    return None

def gitcommit():
    """
    return:
        No git or no commits yet:
            None
        There are git and log not empty:
            <commit sha>
    """
    process = subprocess.Popen("git log | head -n 1",shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    out, err = process.communicate()

    if err=='':
        str1 = "fatal: bad default"
        if out[:len(str1)]==str1:
            #"NO LOG YET"
            return None
        else:
            return out[7:].strip()
    else:
        #"NO GIT REPO"
        print None

    return None
