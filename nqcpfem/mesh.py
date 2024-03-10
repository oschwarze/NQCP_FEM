"""
Model for handling everything related to meshes, including: Import from COMSOL
"""
import gmsh
import meshio
import numpy as np
import re
from typing import Dict


def comsol_to_gmsh(comsol_filename:str,gmsh_filename:str,scale_factor=1,return_raw=False,fix_z=None) -> Dict[str,np.array]:
    """
    Loads a comsol spreadsheet file and saves the resulting mesh as a gmsh file 
    and returns any quantities evaluations that are saved in the spreadsheet file as a dict with the quantity name as key
    and the values in a numpy array

    scale_factor: float specifying how to scale the values of the mesh improted (if they should be converted to different units)
    """

    vertices = [] # vertices in the file
    triangles = [] # triangles in the file
    values = {} # data values

    with open(comsol_filename,'r') as f:
        
        regex = re.compile("(-?\d+\.?\d*E?-?\d*)")
        value_reg = re.compile("\((.*)\)") #extracts the data value name
        # open file and read
        current_list = None
        for l in f.readlines():
            if l[0] =='%':
                if l == '% Coordinates\n':
                    current_list = vertices
                elif l == '% Elements (tetrahedra)\n' or l == "% Elements (triangles)\n": 
                    current_list = triangles
                elif l[:6] == "% Data":
                    value_name = value_reg.findall(l)[0] 
                    values[value_name] = [] # create list in dict
                    current_list = values[value_name]
                continue # skip to next line
            d = [float(r) for r in regex.findall(l)]
            #if len(d) not in Len:
                #Len.append(d)
            current_list.append(d)

        vertices = np.array(vertices)*scale_factor
        if fix_z is not None:
            vertices[:,2] = fix_z

        mesh = meshio.Mesh(
        vertices,
        [("triangle",np.array(triangles)-1)], # COMSOL index starts from 1 so shift the indices
        point_data = values)
        meshio.gmsh.write(gmsh_filename,mesh)
        if return_raw:
            return {k:np.array(v) for k,v in values.items()},vertices,triangles
            
        return {k:np.array(v) for k,v in values.items()}
