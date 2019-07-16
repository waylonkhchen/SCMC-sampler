#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:32:22 2019

@author: waylonchen
"""
##numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
##rx = re.compile(numeric_const_pattern, re.VERBOSE)
##rx.findall("Some example: Jr. it. was - 0.23 between 2.3 and 42.31 seconds")

import re
from constraints import Con



#get algebraic exprs
expression = "x[0] - 0.0004 + 0.9 >= 0.0"
a = expression.split(">=")[0].strip()
print(a)

#count number of x
print(a.count("x"))

#split with + and - but keep them
b = list(filter(None,re.split("([\+\-])", a)))
print(f"b: {b}")


#combine 
c=[]
i=0
while i < len(b):
    if b[i] in "+-": 
        print(i)
        print(b[i] in "+-")
        c.append(b[i]+b[i+1])
        i+=2
        continue
    c.append(b[i])
    i+=1
    
print(c)


def get_sign(s):
    sign =re.findall(r'[\+\-]', s)
    if not sign:
        return 1
    sign = sign[0]
    if sign == "-":
        return -1
    else:
        return 1

def get_sign_str(s):
    sign =re.findall(r'[\+\-]', s)
    if not sign:
        return '+'
    sign = sign[0]
    if sign == "-":
        return '-'
    else:
        return '+'

#def remove_
#pattern = re.compile(r'\s+')
#sentence = re.sub(pattern, '', sentence)

bound =0 
for s in c:

    #extract component index , i.e. the i of x_i
    if "x" in s:
        i_comp = int(re.findall(r'\d+', s)[0]) #i_comp is int, in range(n_dim)
        x_sign = get_sign_str(s) # x_sign = '+' or '-'

    else:
    #        constant = re.findall(r"[-+]?\d*\.\d+|\d+",s)
    #        print(constant)
    #        constant = constant[0]
    #        print(constant)
    #        constant = constant.split()
    #        print(constant)
    #        constant = ''.join(constant)
    #        constant = float(constant)
        constant = float(re.findall(r'\d*\.\d+|\d+', s)[0])
        sign = get_sign(s)
        print(f"bound befor {bound}")
        bound += sign*constant
        print(f"bound after {bound}")

        
print([i_comp, x_sign, bound])

