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

from typing import Iterable
import sympy
import numpy as np

__MOMENTUM_NAMES__ = (r'k_{x}',r'k_{y}',r'k_{z}')
__POSITION_NAMES__ = (r'x',r'y',r'z')

Kx,Ky,Kz = sympy.symbols(__MOMENTUM_NAMES__,commutative=False)
X,Y,Z = sympy.symbols(__POSITION_NAMES__,commutative=False)


def expand_term(term):
    """Expand a term into a tuple of elements where the Ks are individual elements i.e. a term A*x*k_x**2 becomes (A,x,k_x,k_x)

    :param term: _description_
    :type term: _type_
    """
    split = []
    if not isinstance(term,Iterable):
        term = term.args
    for t in term:
        if isinstance(t,sympy.Pow) and not t.is_constant(Kx,Ky,Kz):
            split.extend([t.args[0]]*t.args[1])
            
        else:
            split.append(t)
    
    return tuple(split)

def derivative_of_function(symbol,pos_sym):
    
    import re
    if isinstance(symbol,sympy.Mul):
        post_mul = symbol.args[0]
        # chekc the the post_mul is constant:
        if any((a.name[-3:] == '(x)' or a in (Kx,Ky,Kz,X,Y,Z)) for a in post_mul.atoms()):
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
        if isinstance(x,sympy.Mul):
            assert(x.args[0] == -1)
            x = x.args[1]
        if isinstance(x,sympy.Symbol) and x.name[-3:] == '(x)':
            return sympy.symbols(derivative_of_function(x,conjugate_var),commutative=False)
        
        elif isinstance(x,sympy.Pow) and conjugate_var in x.free_symbols:
            return sympy.diff(x,conjugate_var)
        elif x == conjugate_var:
            return sympy.sympify(1)
        else:
            return 0
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
    
    return term,additional_terms

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
        new_term,add_terms = permute_factors(term,start_i,target_i)
        term = new_term
        # perform recursion to get the remaining expressions into the right target signature (just remove the target_i's True)
        additional_target_signature = target_signature.copy()
        
        if signature_reduction_direction == 'left':
            additional_target_signature.remove(1)
        else:
            additional_target_signature.reverse()
            additional_target_signature.remove(1)
            additional_target_signature.reverse()
        additional_terms.extend([arange_ks(a,additional_target_signature) for a in add_terms])
        current_signature = [not t.is_constant(Kx,Ky,Kz) for t in term]
    return [term,additional_terms]

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
            
            res = sympy.sympify(0)
            for t in terms:
                res = res + rearanger(t)
        
            rearanged_elements.append(res)
    
    arranged_array=sympy.Array(rearanged_elements).reshape(array.shape)
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

def decompose_as_polynomial(expr):
    """Takes an expression which does not contain Kx,Ky or Kz and decomposes it into a tuple describing 
    each terms order wrt the variables X,Y,Z and all functions present and the constant coefficient in front.
    This is usefull for when evaluating the position operators 

    :param expr: expression to decompose. Must not contain Kx,Ky or Kz (since we require it to be commutative)
    :type term: Any
    """
    pass
    
def k_component_signature(term):
    """Computes the component wise signature of the term wrt. the Ks i.e. kx*ky*kz becomes (0,1,2) and kx*kx*kz*kx becomes (0,0,2,0)

    :param term: The term to determine the signature of
    :type term: List[Any]
    """
    term = expand_term(term) # make sure input is of right form
    
    signature = []
    Ks = (Kx,Ky,Kz)
    for t in term:
        if t in Ks:
            signature.append(Ks.index(t))
    
    return signature

