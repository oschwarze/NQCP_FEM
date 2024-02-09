"""
Functions used to import and treat strain maps from COMSOL
"""
import numpy as np 
from typing import Tuple,Callable
import re
import scipy

def import_COMSOL(filename,) -> Tuple[np.ndarray,np.ndarray]:
    """
    Import a comsol strain map from a savefile. The file must contain lines describing the value of the strain tensor components at different points.
    Each line must contain a total of 8 numbers (floats written in scientific notation with E as the symbol for the order of magnitude): 
    the first 2 must be the x and y coordinate of a point and the remaining 6 numbers are the strain tensor components at that point. 
    These are ordered as (x,y,exx,eyy,ezz,exy,exz,eyz)
    """

    position_arr = []
    epsilon_components = []
    regular_expr = re.compile("(-?\d+\.?\d*E?-?\d*)")
    with open('StrainExportSpreadsheet','r') as f:
        for i,l in enumerate(f.readlines()):
            if l[0] != '%': #drop comment lines
                #print([i for i in l])
                nums= [float(r.group(0)) for r in regular_expr.finditer(l)]
                position_arr.append(nums[:2])
                if len(nums) != 8:
                    print(i)
                    print(l)
                    print(nums)
                    print(regular_expr.findall(l))

                epsilon_components.append(nums[2:])

    position_arr = np.array(position_arr)       
    epsilon_components = np.array(epsilon_components)       

    return position_arr,epsilon_components



def interpolate_strain(position_array:np.ndarray,epsilon_components:np.ndarray) -> Callable:
    """
    Given a set of points and values of the strain tensor, construct a function (piecewise linear) that interpolates between these points.
    """
    interpolated = scipy.interpolate.LinearNDInterpolator(position_array,epsilon_components)

    return interpolated