"""This module Contains functions that make working with symbolic representaitons of Hamiltonians easier.
We use the convention, that the wave-vector variables are denoted k_{i} for i = x,y,z, position variables are x,y,z and
functions that depende on x,y,z have a name which ends with `(x)` . All these symbols are non-commuting. 
We often have to take partial derivatives of the functions wrt position variables. These partial derivatives are denoted as another non-commutative symbol
with a subscrpit : _{(x,x,...,x,y,y,...,y,z,z,....,z)}  (always in this order). 
All other symbols are treated as constant.


Naming conventions: 
    - term: an expression involving only multiplication operations
    - expr: an expression possibly involving addition as well
    - Kx,Ky,Kz: momentum symbols
    - X,Y,Z: position symbols
    - symbols ending in `(x)`: symbols representing functions of X,Y and or Z
"""

from operator import index
from typing import Iterable
import sympy
import numpy as np


__MOMENTUM_NAMES__ = (r'k_{x}',r'k_{y}',r'k_{z}')
__POSITION_NAMES__ = (r'x',r'y',r'z')

Kx,Ky,Kz = sympy.symbols(__MOMENTUM_NAMES__,commutative=False)
X,Y,Z = sympy.symbols(__POSITION_NAMES__,commutative=False)


def expand_term(term,split_pow=True):
    """Expand a term into a tuple of elements where the Ks are individual elements i.e. a term A*x*k_x**2 becomes (A,x,k_x,k_x)

    :param term: _description_
    :type term: _type_
    """
    split = []
    if isinstance(term,(sympy.Symbol,sympy.Number)):
        return (term,) # term is just a symbol so we cannot split it up
    if not isinstance(term,Iterable):
        if isinstance(term,sympy.Mul):
            term = term.args
        elif isinstance(term,sympy.Pow):
            term = (term,)
        elif isinstance(term,sympy.Piecewise):
            # splitting up piecewise sucks
            #return (term,)
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"Didn't know how to split up term: {term} ({type(term)})")

    for t in term:
        #print(t,isinstance(t,sympy.Pow)),#t.args[0] in (Kx,Ky,Kz))
        if split_pow and ( isinstance(t,sympy.Pow) and t.args[0] not in (X,Y,Z)+(sympy.symbols('x,y,z'))):
            split.extend([t.args[0]]*t.args[1])
        else:
            split.append(t)
    
    return tuple(split)

def derivative_of_function(symbol,pos_sym):
    
    import re
    if isinstance(symbol,sympy.Mul):
        post_mul = symbol.args[0]
        # chekc the the post_mul is constant:
        if any((a.name[-3:] == '(x)' or a in (Kx,Ky,Kz,X,Y,Z)) for a in post_mul.atoms() if isinstance(a,sympy.Symbol)):
            raise TypeError(f'function symbol was not symbol, symbol times constant or complex conjugate of symbol: {symbol}')
        symbol = symbol.args[1]
    else:
        post_mul  = None
    if isinstance(symbol,sympy.conjugate):
        conjugate = True
        symbol = symbol.args[0]
    else:
        conjugate=False
    
    symbol_name = symbol.name[:-3]
    if symbol_name[-1] != '}':
        name = symbol_name + '_{('+pos_sym.name+')}(x)'
    else:
        derivatives = re.search(r'_\{\((.*)\)\}',symbol_name).group(1)
        Xs = derivatives.count('x')
        Ys = derivatives.count('y')
        Zs = derivatives.count('z')
        
        if pos_sym.name == 'x':
            Xs +=1
        elif pos_sym.name == 'y':
            Ys +=1
        else:
            Zs +=1
        bare_name = re.search('(.*)_{',symbol_name).group(1)
        subscript = ('').join(['x']*Xs+['y']*Ys+['z']*Zs)
        name = bare_name + '_{('+subscript+')}(x)'
    
    derivative = sympy.symbols(name,commutative=False)
    if conjugate: # conjugat before mul to avoid conjugating other factor as well
        derivative = sympy.conjugate(derivative)
    if post_mul is not None:
        derivative = post_mul*derivative
        
    return derivative 

def commutator_map(k):
    """
    The function `commutator_map` returns a commutator function which maps simple expressions, V, to their commutator [k,V].
    if V is a symbolic expression the commutator is computed analytically. If V is a function (symbol with name ending with `(x)`) A new symbol representig its derivative is returned.
    Certain simple extensions of function-symbols are also possible such as:
    - V = complex_conjugate(f(x))
    - V = A*f(x) for A being a constant/symbol independent of X,Y,Z or Kx,Ky,Kz
    
    :param k: The parameter `k` is a variable that can take on the values `Kx`, `Ky`, or `Kz`
    :return: The function `commutator_map` returns a function `commutator`.
    """
    conjugate_var = X if k == Kx else (Y if k == Ky else Z)
    def commutator(x):
        x=sympy.sympify(x)
        if any(a.name[-3:] == '(x)' for a in x.atoms() if isinstance(a,sympy.Symbol)):
            return derivative_of_function(x,conjugate_var)
        else:
            return sympy.diff(x,conjugate_var)
    return commutator
    
def permute_factors(term,start_i,end_i):
    current_i = start_i
    additional_terms = []
    commutators= commutator_map(term[current_i])
    while current_i != end_i:
        direction = np.sign(end_i-current_i) # determine which way to step
        this = term[current_i]
        neighbor = term[current_i+direction]
        if commutators(neighbor) != 0:
            added_term = list(term)
            added_term[current_i] = direction*commutators(neighbor) # add the commutator  (defined as AB = BA + [A,B]) so moving A to the right is + [A,B] whicle moving it to left is -[A,B]
            added_term.pop(current_i+direction) # drop the neighboring term
            additional_terms.append(added_term)
        # we can just permute them 
        new_term = list(term)
        new_term[current_i] = neighbor
        new_term[current_i+direction] = this
        term = new_term
        current_i = current_i + direction
    
    return [term] + additional_terms

def arange_ks(term,target_signature,signature_reduction_direction='left'):
    #print(term,target_signature,symbols)
    current_signature = [not t.is_constant(Kx,Ky,Kz) for t in term]
    additional_terms = []
    while current_signature != target_signature:
        difference = [b-a for b,a in zip(target_signature,current_signature)]
        #print(term,target_signature,current_signature)
        # take the index of the first -1 you see as the start put it into the first +1 you see:
        start_i = difference.index(-1)
        target_i = difference.index(1)
        terms= permute_factors(term,start_i,target_i)
        term = terms[0] # First term is always the one with highest K-order
        add_terms = terms[1:]
        # perform recursion to get the remaining expressions into the right target signature (just remove the target_i's True)
        additional_target_signature = target_signature.copy()
        
        if signature_reduction_direction == 'left':
            additional_target_signature.remove(1)
        else:
            additional_target_signature.reverse()
            additional_target_signature.remove(1)
            additional_target_signature.reverse()
        for t in add_terms:
            additional_terms.extend(arange_ks(t,additional_target_signature))
        current_signature = [not t.is_constant(Kx,Ky,Kz) for t in term]
    return [term] + additional_terms

def construct_target_signature(term,signature_type):
    """
    The function `construct_target_signature` takes a signature type and a term as input, expands the
    term, and returns a target signature based on the signature type.
    
    :param signature_type: The parameter "signature_type" is a string that determines the type of target
    signature to construct. It can take one of the following values:
    - `all left`: All ks are moved to the left
    - `all right`: all ks are moved to the right
    - `FEM`: rightmost K is moved al the way to the right and the other are moved all the way to the left.
    :param term: The term is a list of variables, where each variable can be Kx, Ky, or Kz
    :return: The function `construct_target_signature` returns a list of boolean values. The list describes which factors in the rearanged term should be a k-operator
    """
    term = expand_term(term)
    Nks = sum((v in (Kx,Ky,Kz) for v in term))
    if signature_type == 'all left':
        return [True]*Nks + [False]*(len(term)-Nks)
    elif signature_type == 'all right':
        return  [False]*(len(term)-Nks)+[True]*Nks 
    elif signature_type == 'FEM':
        return [True]*(Nks-1) + [False]*(len(term)-Nks) + [True]
    
def arange_ks_array(array,signature_type,signature_reduction_direction='left'):
    """Given a sympy array containing expressions as entries, rearranges all the elements to acieve the correct signature

    
    
    :param array: The array to rearrange
    :type term: sympy.Array
    :param target_signature: What type of k signature to establish
    :type target_signature: str. see `construct_target_signature` for allowed values 
    :param signature_reduction_direction: _description_, defaults to 'left'. See `arange_Ks` for details
    :type signature_reduction_direction: str, optional
    """
    
    
    # alias for aranging the individual terms according the the signature type and reduction direction
    rearanger = lambda t: arange_ks(t,construct_target_signature(t,signature_type),signature_reduction_direction)
    rearanged_elements = []
    for i,val in enumerate(np.array(array).ravel()):
            # split up value into terms 
            val = sympy.sympify(val).expand()
            terms = val.args if isinstance(val,sympy.core.add.Add) else (val,)
            

            # sum over term in terms: sum over terms that compe from permuting. Mul is to get single expression from list of factors
            res = sympy.Add(*(sympy.Add(*( sympy.Mul(*tt) for tt in rearanger(expand_term(t)))) for t in terms))
            
            
            rearanged_elements.append(res)
    for f in rearanged_elements:
        print(f)
        print(type(f))
        
    arranged_array=sympy.Array(rearanged_elements).reshape(*array.shape)
    return arranged_array
            
def extract_k_independent_part(term):
    """given a term containing ks (either in the begining, the end or both). returns the 'middle' of the term without any ks
    :param term: the term to extract k-independent parts from
    :type term: Iterable[Any]
    """
    
    is_k_dep = [not t.is_constant(Kx,Ky,Kz) for t in term]
    if all(is_k_dep):
        return []
    start_i = is_k_dep.index(False)
    
    middle = term[start_i:]
    is_k_dep = is_k_dep[start_i:]
    if any(is_k_dep):
        middle = term[start_i:start_i+is_k_dep.index(True)]
        is_k_dep = is_k_dep[:is_k_dep.index(True)]
    if any(is_k_dep):
        raise ValueError(f'there were multiple k-independent parts in the expression: {term}')

    return middle

def sort_out_polynomials(expression,symbols):
    """
    The function `sort_out_polynomials` takes an expression and a list of symbols as input, and returns
    a list of tuples representing the terms of the polynomial expression, sorted according to the
    ordering of the symbols.
    
    :param expression: The `expression` parameter is the polynomial expression that you want to sort
    out. It can be any valid polynomial expression in terms of the given symbols
    :param symbols: The `symbols` parameter is a list of symbols that represent the variables in the
    polynomial expression. These symbols should be commutative in order for the expression to be treated as a polynomial in them.
    :return: The function `sort_out_polynomials` returns a list of tuples. Each tuple contains two
    elements: the first element is a tuple representing the orders of the polynomial terms (sorted similarly to `symbols`), and the
    second element is the corresponding coefficient. If a order is None, the corresponding coefficient will depend on the related symbol in a non-polynomial manner.
    """
    if any(not s.is_commutative for s in symbols):
        raise TypeError(f'symbols have to be commutative in order be treated as polynomials: the following symbols were non-commutative: {[s for s in symbols if not s.is_commutative]}')
    
    
    from sympy import polys
    exprs = ((tuple(),expression),) # each element is a term in the polynomial represented by tuple containin tuple of orders and the coefficient. orders are ordered according to ordering of symbols. 
    symbols_list = list(symbols)
    while len(symbols_list):
        sym = symbols_list.pop(0)
        new_exprs = []
        for expr in exprs:
            ex = expr[1]
            if not ex.is_polynomial(sym):
                if isinstance(ex,sympy.Add):
                    # remove the parts that are not polynomials
                    not_poly = sympy.Add(*(a for a in ex.args if not a.is_polynomial(sym)))
                    is_poly = sympy.Add(*(a for a in ex.args if a.is_polynomial(sym)))
                    new_exprs.append((expr[0]+(None,),not_poly))
                    print(not_poly,is_poly)
                    ex = is_poly
                else:
                    new_exprs.append((expr[0]+(None,),ex)) # not polynomial # not polynomial
            
            if ex.is_constant(sym):
                new_exprs.append((expr[0]+(0,),ex))
                continue 
            # treat all the expressions in expres as univariate polynomial in sym (and all other symbols as constants)
            
            
            univariate_expr = polys.Poly(ex,sym)#domain=domain)
            new_exprs.extend(((expr[0]+t[0],t[1]) for t in univariate_expr.all_terms()))
        exprs = new_exprs
        
    return [ e for e in exprs if e[1]!=0]
        

def extract_valid_bipartition_part(expr):
    """Extracts the bipartition-parts of an expression for which there exists analytically solved matrix representations
    :param expr: _description_
    :type expr: sympy.Piecewise
    
    
    Returns as list of all the atomic piecewise functions (1 if X_i>0, 0 else ) for X_i = x,y,z, as well as the remaining factor 
    """
    xx,yy,zz = sympy.symbols('x,y,z') # commuting ones so that we can use the min the relations
    atomic_relations = ((xx>0,xx>=0),(yy>0,yy>=0),(zz>0,zz>=0))

    valid_relations = {r[0] for r in atomic_relations}
    valid_relations.update(r[1] for r in atomic_relations)
    for i in range(3):
        j = (i+1)%3
        for e in range(2):
            for f in range(2):
                valid_relations.add(sympy.And(atomic_relations[i][e],atomic_relations[j][f]))

    for i in range(2):
        for j in range(2):
            for k in range(2):
                valid_relations.add(sympy.And(atomic_relations[0][i],atomic_relations[1][j],atomic_relations[2][k]))

    valid_bipartitions = []
    remains = None
    position_symbols = (xx,yy,zz)
    
    expr = expr.subs({X:xx,Y:yy,Z:zz})
    
    if len(expr.args)>2:
        raise NotImplementedError(f'too complex PieceWise: {expr}')
    
    for arg in expr.args:
        if isinstance(arg[1], sympy.logic.boolalg.BooleanTrue):
            continue
        if arg[1] in valid_relations:
            present_syms = arg[1].free_symbols.intersection(set(position_symbols))
            valid_syms = [p  for p in present_syms if p not in arg[0].free_symbols]
            for p in valid_syms:
                # evaluate the bipartition directions analytically
                valid_bipartitions.append(sympy.Piecewise((1,p>0),(0,True)))
                
                remains = arg[0]
            else:
                remains = expr.subs({p:1 for p in valid_syms}) # ignore all peacewise for the valid ones
        else:
            remains = expr
        print(valid_bipartitions,remains)       
    return valid_bipartitions,remains
                

def k_component_signature(term,k_symbols=None):
    """Computes the component wise signature of the term wrt. the Ks i.e. kx*ky*kz becomes (0,1,2) and kx*kx*kz*kx becomes (0,0,2,0)

    :param term: The term to determine the signature of
    :type term: List[Any]
    """
    term = expand_term(term) # make sure input is of right form
    if k_symbols is None:
        k_symbols = (Kx,Ky,Kz)
    signature = []
    for t in term:
        if t in k_symbols:
            signature.append(k_symbols.index(t))
    return signature

    
def dummify_non_commutative(expr):
    """
    The function dummify_non_commutative takes an expression as input and returns a dictionary of dummy
    symbols with the same assumptions as the free symbols in the expression.
    
    :param expr: The `expr` parameter is the expression for which we want to dummify the non-commutative
    symbols
    :return: The function `dummify_non_commutative` returns a dictionary where the keys are the free
    symbols in the input expression `expr`, and the values are dummy symbols with the same assumptions
    as the corresponding free symbols.
    """
    return {s:sympy.Dummy(**s.assumptions0) for s in expr.free_symbols}

    
def symbolize_array(array,symbol_base_name):
    """replaces expressions in the array by abstract symbols in an efficient manner (with as few new symbols as possible)
    It is assumed the that array has been 'symbolized' when X,Y,Z and x,y,z no longer appear in the array

    :param array: sympy array to symbolize
    :type array: Sympy.Array
    :param symbol_base_name: str: bas
    :type symbol_base_name: _type_
    """

    if isinstance(symbol_base_name,sympy.Symbol):
        symbol_base_name = symbol_base_name.name
    
    if symbol_base_name[-3:] == '(x)':
        symbol_base_name = symbol_base_name[:-3]
    
    def new_symbol():
        index_counter = 0 
        while True:
            index_counter +=1
            next_name = '('+symbol_base_name+f')_{index_counter}(x)'
            yield sympy.Symbol(next_name,commutative=False)
    
    symbol_generator = new_symbol()
    
    symbol_dict = {}
    new_array = array.as_mutable()
    for i,expr in enumerate(np.array(new_array).ravel()):
        #check if we have seen a similar_expression:
        if expr.is_constant():
            continue # no need to do anything
        
        post_fix = None # post_fix function is applied to the EXPRESSION IN THE DICT and should become the EXPRESSION IN THE ARRAY
        if expr in symbol_dict.values():
            post_fix = lambda x:x
        elif -expr in symbol_dict.values():
            post_fix = lambda x: -1*X
        elif sympy.conjugate(expr) in symbol_dict.values():
            post_fix = lambda x: sympy.conjugate(x)
        elif any((expr/f).is_constant(X,Y,Z,*sympy.symbols('x,y,z')) for f in symbol_dict.values()):
            for f in symbol_dict.values():
                coeff = expr/f
                if coeff.is_constant(X,Y,Z,*sympy.symbols('x,y,z')):
                    post_fix = lambda x:coeff*x
                    break
            else:
                ValueError('we should not get here...')
            
        ix = np.unravel_index(i,array.shape)
        # replace with symbol
        if post_fix is None:
            sym = next(symbol_generator)
            symbol_dict[sym] = expr
            new_array[ix] = sym
        else:
            # we have already seen a similar_expression
            sym = next(s for s,f in symbol_dict.items() if post_fix(f)==expr)
            new_array[ix] = post_fix(sym)
    return sympy.Array(new_array),symbol_dict 