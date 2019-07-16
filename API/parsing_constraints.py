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
#from constraints import Constraint


def get_simple_boundaries():
    pass



def filter_simple_constraints(constraints_list, var_name = "x"):
    """separate simple contraints from complicated in the list of contraints 
    a constraint is simple when only one variable contained => can become sampling bounds
    
    Parameters
    ----------
    constraints_list:   List,
        containing string that represents the constraints, e.g. 'x[0] + x[1] - 0.5 >= 0 "
    var_name: str
        name of the variable used, default is 'x'
        
    Returns
    -------
    simpl, compl: List, containing simple or complicated constraints
    """
    simpl, compl = [],[]
    for expr in constraints_list:
        if expr.count("x") ==1: #if only one variale, it's a simple constraint 
            simpl.append(expr)
        else:
            compl.append(expr)
    return simpl, compl

def combine_sign_w_var(expr_list):
    """combine sign with variable x[i] or numeric
        e.g.  ['0.0035 ', '-', ' x[0]'] -> ['0.0035 ', '- x[0]']
    Parameters
    ----------
    expr_list: List,
        a list that consists of one single simple expression
        
    Returns
    -------
    c: List,
        
        
    """
    c=[]
    i=0
    while i < len(expr_list):
        if expr_list[i] in "+-": 
            c.append(expr_list[i]+expr_list[i+1])
            i+=2
            continue
        c.append(expr_list[i])
        i+=1
    return c


def parsing_simpl(simpl):
    """
    
    """
    #get algebraic only
    simpl = simpl.split(">=")[0].strip()
    #split with + or -
    simpl = list(filter(None,re.split("([\+\-])", simpl)))
    #combine sign with variable
    simpl = combine_sign_w_var(simpl)
    return simpl

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

    
def convert_parsed_to_bound(simpl, var_name = "x"):         
    bound =0 
    for s in simpl:
    
        #extract component index , i.e. the i of x_i
        if var_name in s:
            i_comp = int(re.findall(r'\d+', s)[0]) #i_comp is int, in range(n_dim)
            x_sign = get_sign_str(s) # x_sign = '+' or '-'    
        else:
            constant = float(re.findall(r'\d*\.\d+|\d+', s)[0])
            sign = get_sign(s)
            bound += sign*constant
    return [x_sign, i_comp, bound]






#def remove_
#pattern = re.compile(r'\s+')
#sentence = re.sub(pattern, '', sentence)


#def constraints2bounds(file_name = "../alloy.txt"):
#    """
#    """
#    cons1 = Constraint(file_name)
#    exprs_list = cons1.get_exprs_list()
#    #separate constraints
#    simpl, compl = filter_simple_constraints(exprs_list)
#    
#    bounds = []
#    for expr  in simpl:
#        parsed = parsing_simpl(expr)
#        bounds.append(convert_parsed_to_bound(parsed))
#    return bounds








##get algebraic exprs
#expression = AA[3]
#a = expression.split(">=")[0].strip()
#print(a)
#
##count number of x
#print(a.count("x"))
#
##split with + and - but keep them
#b = list(filter(None,re.split("([\+\-])", a)))
#print(f"b: {b}")
#
#
##combine 
#c=[]
#i=0
#while i < len(b):
#    if b[i] in "+-": 
#        print(i)
#        print(b[i] in "+-")
#        c.append(b[i]+b[i+1])
#        i+=2
#        continue
#    c.append(b[i])
#    i+=1
#    
#print(c)
#
#
#bound =0 
#for s in c:
#
#    #extract component index , i.e. the i of x_i
#    if "x" in s:
#        i_comp = int(re.findall(r'\d+', s)[0]) #i_comp is int, in range(n_dim)
#        x_sign = get_sign_str(s) # x_sign = '+' or '-'
#
#    else:
#    #        constant = re.findall(r"[-+]?\d*\.\d+|\d+",s)
#    #        print(constant)
#    #        constant = constant[0]
#    #        print(constant)
#    #        constant = constant.split()
#    #        print(constant)
#    #        constant = ''.join(constant)
#    #        constant = float(constant)
#        constant = float(re.findall(r'\d*\.\d+|\d+', s)[0])
#        sign = get_sign(s)
#        print(f"bound befor {bound}")
#        bound += sign*constant
#        print(f"bound after {bound}")
#
#        
#print([i_comp, x_sign, bound])

