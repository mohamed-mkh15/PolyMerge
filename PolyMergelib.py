# Install PyTorch
"""
pip install torch==1.12.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# Install MMCV
pip install openmim
mim install mmcv-full==1.6.0

pip install pyquaternion
"""

import mmcv
import cv2

import math
import numpy as np
import pandas as pd
from pyquaternion import Quaternion

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.size"] = 8


import warnings
warnings.filterwarnings('ignore')



## Merge functions 
def vector_proj(p1, p2, p3): 
    """
    Computes the vector projection of point p1 onto the line defined by points p2 and p3.
    
    Parameters:
    - p1: A numpy array representing the coordinates of point p1.
    - p2: A numpy array representing the coordinates of the first point defining the line.
    - p3: A numpy array representing the coordinates of the second point defining the line.

    Returns:
    - p4: A numpy array representing the coordinates of the closest point on the line (p2-p3) to point p1.

    Note:
    - If point p1 coincides with either point p2 or p3, point p1 is returned.
    Adapted from: https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point
    """
    # Check if p1 coincides with either p2 or p3
    if all(p1 == p2) or all(p1 == p3):
        return p1
    else:
        # Calculate vector from p2 to p1 and p2 to p3
        p2p1 = p1 - p2
        p2p3 = p3 - p2
        # Calculate the scalar projection of p2p1 onto p2p3
        t = np.dot(p2p1, p2p3) / np.dot(p2p3, p2p3)
        # Calculate the closest point on the line to point p1
        p4 = p2 + t * p2p3
        return p4 
         
def vector_proj_in(p1, p2, p3): 
    """
    Computes the vector projection of point p1 onto the line segment defined by points p2 and p3.

    Parameters:
    - p1: A numpy array representing the coordinates of point p1.
    - p2: A numpy array representing the coordinates of the first endpoint of the line segment.
    - p3: A numpy array representing the coordinates of the second endpoint of the line segment.

    Returns:
    - p4_in: A numpy array representing the coordinates of the closest point on the line segment (p2-p3) to point p1.

    Note:
    - If point p1 coincides with either point p2 or p3, point p1 is returned.
    - If the projection point is outside the line segment, it's clamped to the closest endpoint.
    - Adapted from: https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point
    """
    if all(p1 == p2) or all(p1 == p3):
        return p1
    else:
        p2p1 = p1 - p2
        p2p3 = p3 - p2
        t = np.dot(p2p1, p2p3) / np.dot(p2p3, p2p3)
        # if you need the the closest point belonging to the segment
        t_in = max(0, min(1, t))
        p4_in = p2 + t_in * p2p3
        return p4_in 
   
def checkOrtho(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Checks whether two line segments are orthogonal (perpendicular) to each other.

    Parameters:
    - x1, y1: Coordinates of the first endpoint of the first line segment.
    - x2, y2: Coordinates of the second endpoint of the first line segment.
    - x3, y3: Coordinates of the first endpoint of the second line segment.
    - x4, y4: Coordinates of the second endpoint of the second line segment.

    Returns:
    - True if the two line segments are orthogonal, False otherwise.

    Note:
    - Each line segment is defined by its two endpoints.
    """
    # Both lines have infinite slope
    if (x2 - x1 == 0 and x4 - x3 == 0):
        return False
    # Only line 1 has infinite slope
    elif (x2 - x1 == 0):
        m2 = (y4 - y3) / (x4 - x3)
 
        if (m2 == 0):
            return True
        else:
            return False
    # Only line 2 has infinite slope
    elif (x4 - x3 == 0):
        m1 = (y2 - y1) / (x2 - x1)
 
        if (m1 == 0):
            return True
        else:
            return False
    # Find slopes of the lines
    else:         
        m1 = (y2 - y1) / (x2 - x1)
        m2 = (y4 - y3) / (x4 - x3)
        # Check if their product is -1
        if (m1 * m2 == -1):
            return True
        else:
            return False

def get_p2p3p4(p, poly):
    """
    Gets the closest segment of a given polyline to point p.

    Inputs:
    - p: A numpy array representing the coordinates of point p.
    - poly: NumPy array representing the vertices of the polyline.

    Outputs:
    - p2_in: A numpy array representing the coordinates of the first endpoint of the closest line segment to point p.
    - p3_in: A numpy array representing the coordinates of the second endpoint of the closest line segment to point p.
    - p4: A numpy array representing the coordinates of the closest point on the extension of the closest line segment to point p.
    or p4_in: A numpy array representing the coordinates of the closest point on the closest line segment to point p.
    """
    min_dist = np.inf
    min_dist_in = np.inf
    # Iterate through each line segment defined by consecutive pairs of points in the polyline
    for p2_temp, p3_temp in zip(poly[:-1], poly[1:]):
        # Calculate projection points using both vector_proj and vector_proj_in functions
        p4_temp = vector_proj(p, p2_temp, p3_temp)
        p4_in_temp = vector_proj_in(p, p2_temp, p3_temp)
        # Calculate distances between point p and projection points
        dist = math.dist(p, p4_temp)
        dist_in = math.dist(p, p4_in_temp)

        # Update minimum distances and corresponding points if necessary
        if dist < min_dist:
            min_dist = dist
            p2 = p2_temp
            p3 = p3_temp
            p4 = p4_temp
        
        if dist_in < min_dist_in:
            min_dist_in = dist_in
            p2_in = p2_temp
            p3_in = p3_temp
            p4_in = p4_in_temp
            
    # Decide which point to take (p4) or (p4_in) based on the in_line projection point (p4_in)
    if all(p4_in == p2) or all(p4_in == p3):
        # If p4_in coincides with p2 or p3, use projection on the extension of the polyline
        p4 = vector_proj(p, p2_in, p3_in)
        return p2_in, p3_in, p4
    else: 
        return p2_in, p3_in, p4_in

def get_p2p3p4_in(p, poly):
    """
    Gets the closest line segment of a given polyline to point p.

    Inputs:
    - p: A numpy array representing the coordinates of point p.
    - poly: NumPy array representing the vertices of the polyline.
    
    Outputs:
    - p2_in: A numpy array representing the coordinates of the first endpoint of the closest line segment to point p.
    - p3_in: A numpy array representing the coordinates of the second endpoint of the closest line segment to point p.
    - p4_in: A numpy array representing the coordinates of the closest point on the closest line segment to point p.

    Note:
    - Returns the closest projection point (p4_in) and the line segment endpoints (p2_in, p3_in).
    """
    min_dist_in = np.inf
    # Iterate through each line segment defined by consecutive pairs of points in the polyline
    for p2_temp, p3_temp in zip(poly[:-1], poly[1:]):
        # Calculate projection points using vector_proj_in function
        p4_in_temp = vector_proj_in(p, p2_temp, p3_temp)
        # Calculate distance between point p and in-line projection point
        dist_in = math.dist(p, p4_in_temp)
        # Update minimum distance and corresponding points if necessary
        if dist_in < min_dist_in:
            min_dist_in = dist_in
            p2_in = p2_temp
            p3_in = p3_temp
            p4_in = p4_in_temp
    return p2_in, p3_in, p4_in
    
def projection_pairs(poly1, poly2):
    """
    Calculates the projections of all points of poly1 on poly2 and returns these pairs.

    Inputs:
    - poly1: NumPy array representing the vertices of  the first polyline.
    - poly2: NumPy array representing the vertices of the second polyline.

    Outputs:
    - projection_points: A numpy array containing pairs of points [[p1,p2,p3,p4], ...].

    Note:
    - Uses the get_p2p3p4 function to find the projection points.
    """
    points = []
    # Iterate through each point in poly1
    for p1 in poly1:
        # Get the projection point (p4) on poly2 and the corresponding line segment endpoints (p2, p3)
        p2, p3, p4 = get_p2p3p4(p1, poly2)
        points.append([p1,p2,p3,p4])

    return np.array(points)

def projection_pairs_in(poly1, poly2, S_on_M=True):
    """
    Calculates the in-line projections of all points of poly1 on poly2 and returns these pairs.

    Inputs:
    - poly1: NumPy array representing the vertices of the first polyline.
    - poly2: NumPy array representing the vertices of the second polyline.
    - S_on_M: A boolean indicating whether we are projecting poly1 onto poly2 (True) or vice versa (False).

    Outputs:
    - points: A numpy array containing pairs of points [[p,p2,p3,p4], ...].

    Note:
    - Uses the get_p2p3p4_in function to find the projection points.
    """
    points = []
    # Iterate through each point in poly1
    for p in poly1:
        # Get the projection point (p4) on poly2 and the corresponding line segment endpoints (p2, p3)
        p2, p3, p4 = get_p2p3p4_in(p, poly2)
        # Check if the projection point is on the edge of poly2
        if S_on_M and all(p4==poly2[-1]): # p4 is on the tail of poly2
            edge = True
            p4=p
        elif not S_on_M and all(p4==poly2[0]): # p4 is on the head of poly2
            edge = True
            p4=p
        else: 
            edge=False
        # Append the point and its projection points to the list
        points.append([p,p2,p3,p4])

    return np.array(points)

def point_to_polyline_dist(p1, poly):
    """
    Determines the distance between point p1 and the given polyline.

    Inputs:
    - p1: A numpy array representing the coordinates of point p1.
    - poly: NumPy array representing the vertices of the polyline.

    Outputs:
    - Distance between p1 and its projection on the polyline.

    Note:
    - Uses the get_p2p3p4_in function to find the projection point.
    """
    # Get the projection point (p4) on the polyline and the corresponding line segment endpoints (p2, p3)
    p2, p3 , p4 = get_p2p3p4_in(p1, poly)
    # Calculate the distance between point p1 and its projection point (p4)
    return math.dist(p1, p4)

def polyline_merge_check(poly1, poly2, T):
    """
    Function that checks if the two polylines should be merged or not. 
    Inputs: 
    polyline1, polyline2 ->
    T -> threshold (minimum distance between given point and polyline to determine if they shall be merged or not)

    Outputs:
    check: True (should be merged) / False (shouldn't be merged)
    """
    check = "False"
    for i, p1 in enumerate(poly1):
        dist = point_to_polyline_dist(p1, poly2)
        if dist < T:
            check = "True"
            break
    for j, p2 in enumerate(poly2):
        dist = point_to_polyline_dist(p2, poly1)
        if dist < T:
            check = "True"
            break
    return i,j, check

# Merge polylines function iterations (use  V3). I kept them to show my line of thought
#(Depricated)
def merge_polys_v1(poly1, poly2):
    """
    Merge two polygons into a single polygon.

    Parameters:
    - poly1: NumPy array representing the vertices of the first polygon. Each row contains the (x, y) coordinates of a vertex.
    - poly2: NumPy array representing the vertices of the second polygon. Each row contains the (x, y) coordinates of a vertex.

    Returns:
    - NumPy array representing the vertices of the merged polygon. Each row contains the (x, y) coordinates of a vertex.

    This function merges two polygons, poly1 and poly2, into a single polygon. It starts by calculating projection pairs 
    between poly1 and poly2, and vice versa. Then, it calculates the midpoint between the projections of each polygon's vertices 
    onto the other polygon. The merged polygon initially consists of the midpoint projections of poly2 onto poly1.

    For each midpoint projection of poly1 onto poly2, the function determines its position relative to the edges of the merged polygon. 
    Depending on this position, the midpoint projection is inserted into the merged polygon in one of three scenarios:
    - Scenario 1 (s1): The midpoint projection is between two vertices of the merged polygon.
    - Scenario 2 (s2): The midpoint projection is behind the starting vertex of an edge of the merged polygon.
    - Scenario 3 (s3): The midpoint projection is in front of the ending vertex of an edge of the merged polygon.

    Finally, the function returns the vertices of the merged polygon.
    """
    
    # Calculate projection pairs between poly1 and poly2, and vice versa
    p1pp = projection_pairs(poly1, poly2) # poly1_projection_pairs -> projections of poly1 points on poly2
    p2pp = projection_pairs(poly2, poly1) # poly2_projection_pairs -> projections of poly2 points on poly1
    
    # Calculate the midpoint between projections of each polygon's vertices
    p1ms = (p1pp[:,0]+p1pp[:,3])/2
    p2ms = (p2pp[:,0]+p2pp[:,3])/2
    
    # Initialize the merged polygon with the midpoints of projections from poly2 to poly1
    polyM = p2ms 
    # Iterate over midpoints of projections from poly1 to poly2
    for p1m in p1ms:
        # Get points p2, p3, and p4 based on p1m and the current merged polygon
        p2, p3, p4 = get_p2p3p4(p1m, polyM)

        # check scenario s1 (p1 is between p2-p3), s2 (p1 behind p2), s3 (p1 in front of p3)
        p2p3_u = (p3-p2) / np.linalg.norm(p3-p2)
        p2p4_u = (p4-p2) / np.linalg.norm(p4-p2) 
        p3p4_u = (p4-p3) / np.linalg.norm(p4-p3)

        p2_index = np.argwhere(np.all(polyM == p2, axis=1))[0][0]
        p3_index = np.argwhere(np.all(polyM == p3, axis=1))[0][0]

        if all(np.sign(p2p3_u) == np.sign(-p2p4_u)):
            polyM = np.insert(polyM, p2_index, p1m, axis=0)

        elif all(np.sign(p2p4_u) == np.sign(p3p4_u)):
            polyM = np.insert(polyM, p3_index+1, p1m, axis=0)

        else:
            polyM = np.insert(polyM, p2_index+1,p1m, axis=0)
    return polyM

def merge_polys_v2(poly1, poly2):
    """
    Merge two polygons into a single polygon. With little changes than the first one

    Parameters:
    - poly1: NumPy array representing the vertices of the first polygon. Each row contains the (x, y) coordinates of a vertex.
    - poly2: NumPy array representing the vertices of the second polygon. Each row contains the (x, y) coordinates of a vertex.

    Returns:
    - NumPy array representing the vertices of the merged polygon. Each row contains the (x, y) coordinates of a vertex.

    This function merges two polygons, poly1 and poly2, into a single polygon. It starts by calculating projection pairs 
    between poly1 and poly2, projecting points from poly2 onto poly1. Then, it calculates the midpoint between the projections 
    of each polygon's vertices. The merged polygon initially consists of the midpoints of projections from poly1 to poly2.

    For each midpoint projection of poly2 onto poly1, the function determines its position relative to the edges of the merged polygon. 
    Depending on this position, the midpoint projection is inserted into the merged polygon in one of three scenarios:
    - Scenario 1 (s1): The midpoint projection is between two vertices of the merged polygon.
    - Scenario 2 (s2): The midpoint projection is behind the starting vertex of an edge of the merged polygon.
    - Scenario 3 (s3): The midpoint projection is in front of the ending vertex of an edge of the merged polygon.

    Finally, the function removes any duplicate vertices and returns the vertices of the merged polygon.
    """
    # Calculate projection pairs between poly1 and poly2, and vice versa
    p1pp = projection_pairs_in(poly1, poly2, S_on_M=False) # poly1_projection_pairs -> projections of poly1 points on poly2
    p2pp = projection_pairs_in(poly2, poly1, S_on_M=True) # poly2_projection_pairs -> projections of poly2 points on poly1

    # Calculate the midpoint between projections of each polygon's vertices
    p1ms = (p1pp[:,0]+p1pp[:,3])/2
    p2ms = (p2pp[:,0]+p2pp[:,3])/2

    # Initialize the merged polygon with the midpoints of projections of poly1 on poly2
    polyM = p1ms #merged poly
    for pm in p2ms:
        p2, p3, p4 = get_p2p3p4_in(pm, polyM)
        if all(p4==polyM[-1]):
            p4=pm

        # check scenario s1 (p1 is between p2-p3), s2 (p1 behind p2), s3 (p1 in front of p3)
        p2p3_u = (p3-p2) / np.linalg.norm(p3-p2)
        p2p4_u = (p4-p2) / np.linalg.norm(p4-p2) 
        p3p4_u = (p4-p3) / np.linalg.norm(p4-p3)

        p2_index = np.argwhere(np.all(polyM == p2, axis=1))[0][0]
        p3_index = np.argwhere(np.all(polyM == p3, axis=1))[0][0]

        if all(np.sign(p2p3_u) == np.sign(-p2p4_u)):
            polyM = np.insert(polyM, p2_index, pm, axis=0)

        elif all(np.sign(p2p4_u) == np.sign(p3p4_u)):
            polyM = np.insert(polyM, p3_index+1, pm, axis=0)

        else:
            polyM = np.insert(polyM, p2_index+1,pm, axis=0)
    # Remove duplicate vertices
    polyM = delete_duplicate(polyM)
    return polyM

def merge_polys_v3(poly1, poly2):
    """
    Merge two polygons into a single polygon.

    Parameters:
    - poly1: NumPy array representing the vertices of the first polygon. Each row contains the (x, y) coordinates of a vertex.
    - poly2: NumPy array representing the vertices of the second polygon. Each row contains the (x, y) coordinates of a vertex.

    Returns:
    - NumPy array representing the vertices of the merged polygon. Each row contains the (x, y) coordinates of a vertex.

    This function merges two polygons, poly1 and poly2, into a single polygon. It starts by arranging the vertices of poly1 
    to form the initial merged polygon. The function then iterates over each vertex of poly1 and computes its corresponding 
    projection point (p4) onto poly2. If p4 does not coincide with either endpoint of the line segment (p2, p3) in poly2, 
    the function averages the vertex with p4 and updates poly1. Next, the function iterates over each vertex of poly2, 
    determining its projection onto poly1 and inserting it into the merged polygon based on its position relative to the edges 
    of the merged polygon. If the projection coincides with the starting or ending vertex of the merged polygon, the vertex is 
    inserted accordingly. Otherwise, the midpoint between the vertex and its projection onto the merged polygon is computed 
    and inserted into the merged polygon. Finally, any duplicate vertices are removed, and the merged polygon is returned.
    """
    # Get the first and last vertices of poly1
    polyM_e0 = poly1[0]     # first edge (Head) of main poly
    polyM_e1 = poly1[-1]    # last edge (Tail) of main poly
    # Initialize the merged polygon with poly1
    polyM = poly1 
    # Average the vertices of poly1 that correspond to the projection points (p4) onto poly2
    for i in range(len(polyM)):
        p = polyM[i]
        p2, p3, p4 = get_p2p3p4_in(p, poly2)
        if not(all(p4==p2) or all(p4==p3)):
            polyM[i] = (p+p4)/2
            
    # Iterate over vertices of poly2
    for p in poly2:
        # Get the projection point (p4) of p onto poly1
        p2, p3, p4 = get_p2p3p4_in(p, poly1)
        p2_index = np.argwhere(np.all(polyM == p2, axis=1))[0][0]
        p3_index = np.argwhere(np.all(polyM == p3, axis=1))[0][0]
        
        # Insert p into the merged polygon based on its position relative to poly1
        if all(p4==polyM_e0):
            polyM = np.insert(polyM, p2_index, p, axis=0)
        elif all(p4==polyM_e1):
            polyM = np.insert(polyM, p3_index+1, p, axis=0)
        else:
            # Calculate the midpoint between p and its projection onto the merged polygon
            p2, p3, p4 = get_p2p3p4_in(p, polyM)
            pm = (p+p4)/2
            p2_index = np.argwhere(np.all(polyM == p2, axis=1))[0][0]
            polyM = np.insert(polyM, p2_index+1,pm, axis=0)
    # Remove duplicate vertices
    polyM = delete_duplicate(polyM)
    return polyM

def merge_polys_v4(poly1,poly2):
    """
    Merge two polygons into a single polygon.

    Parameters:
    - poly1: NumPy array representing the vertices of the first polygon. Each row contains the (x, y) coordinates of a vertex.
    - poly2: NumPy array representing the vertices of the second polygon. Each row contains the (x, y) coordinates of a vertex.

    Returns:
    - NumPy array representing the vertices of the merged polygon. Each row contains the (x, y) coordinates of a vertex.

    This function merges two polygons, poly1 and poly2, into a single polygon. It starts by averaging the vertices of poly1 
    that correspond to the projection points (p4) onto poly2. If p4 does not coincide with either endpoint of the line segment 
    (p2, p3) in poly2, the function computes the average of the vertex and p4. The initial merged polygon consists of these averaged 
    vertices and unchanged vertices from poly1. Next, the function iterates over vertices of poly2, determining their projection 
    onto the merged polygon. Depending on the position of the projection relative to the edges of the merged polygon, the vertex 
    is inserted into the merged polygon accordingly. If the projection coincides with the starting or ending vertex of the merged 
    polygon, the vertex is inserted at the corresponding position. Otherwise, the midpoint between the vertex and its projection 
    onto the merged polygon is computed and inserted into the merged polygon. Finally, any duplicate vertices are removed, and 
    the merged polygon is returned.
    It differs from the last one in the vertex averaging approach and insertion starategy
    """
    # Initialize the merged polygon
    polyM = [] 
    # Average the vertices of poly1 that correspond to the projection points (p4) onto poly2
    for i, p in zip(range(len(poly1)),poly1):
        p2, p3, p4 = get_p2p3p4_in(p, poly2)
        p21, p31, p41 = get_p2p3p4_in(p2, poly1)
        if not(all(p4==p2) or all(p4==p3)):
            if not(all(p41==p21) or all(p41==p31)):
                polyM.append(p)
            else:
                polyM.append((p+p4)/2)
        else:
            polyM.append(p)
    polyM = np.array(polyM)
    poly_temp = poly1
    polyM_e0 = poly_temp[0]     # first edge of main poly
    polyM_e1 = poly_temp[-1]    # last edge of main 
    end = False
    
    # Iterate over vertices of poly2
    for p in poly2:
        p2, p3, p4 = get_p2p3p4_in(p, poly_temp)
        p2_index = np.argwhere(np.all(poly_temp == p2, axis=1))[0][0]
        p3_index = np.argwhere(np.all(poly_temp == p3, axis=1))[0][0]
        
        if end==True:
            polyM = np.insert(polyM, -1, p, axis=0)
            poly_temp = np.insert(poly_temp, -1, p, axis=0)
        if all(p4==polyM_e0):
            polyM = np.insert(polyM, p2_index, p, axis=0)
            poly_temp = np.insert(poly_temp, p2_index, p, axis=0)
            olyM_e0 = p
        elif all(p4==polyM_e1):
            polyM = np.insert(polyM, p3_index+1, p, axis=0)
            poly_temp = np.insert(poly_temp, p3_index+1, p, axis=0)
            polyM_e1 = p
            end = True
        else:
            p2, p3, p4 = get_p2p3p4_in(p, poly_temp)
            pm = (p+p4)/2
            p2_index = np.argwhere(np.all(poly_temp == p2, axis=1))[0][0]
            polyM = np.insert(polyM, p2_index+1,pm, axis=0)
            poly_temp = np.insert(poly_temp, p2_index+1,pm, axis=0)
    # Remove duplicate vertices
    polyM = delete_duplicate(polyM)
    return polyM

def refine_token(token, proximity_th):
    """
    Refine a map token by merging connected polylines based on proximity threshold.

    Parameters:
    - token: DataFrame representing the token. It should contain columns: 'vectors', 'scores', and 'labels'. 
             'vectors' column contains polyline vectors, 'scores' column contains scores, and 'labels' column contains labels.
    - proximity_th: Proximity threshold determining when to merge polylines. polylines with a distance less than this threshold 
                    will be considered for merging.

    Returns:
    - DataFrame representing the refined token after merging connected polylines based on the proximity threshold.

    This function refines a token by merging connected polylines based on a proximity threshold. It creates a graph to store 
    the connected polylines (nodes) and merges them at the end. It iterates over each polyline in the token and checks its 
    proximity with other polylines. If the proximity check passes and the polylines have different labels or at least one 
    polyline has label=0 (representing ped_crossing), they are considered for merging. After identifying connected components 
    in the graph, it merges the polylines within each connected component. If the polylines in a component have different 
    labels, the merged polyline is assigned label 2. Finally, the function returns the refined token after merging.

    Example usage:
    refined_token = refine_token(token_df, proximity_threshold)
    """
    # Create a copy of the token
    token_refined = pd.DataFrame() #.copy(deep=True)
    token_refined['vectors']= token['vectors']
    token_refined['scores']= token['scores']
    token_refined['labels']= token['labels']
    # Create a graph to store the connected polylines (nodes) to merge them at the end
    import networkx as nx
    G = nx.Graph()

    # Iterate over each polyline in the token
    for idx, poly, label in zip(token.index, token['vectors'], token['labels']):
        poly2_list = token.drop(idx)
        # Iterate over other polylines
        for idx2, poly2, label2 in zip(poly2_list.index, poly2_list['vectors'], poly2_list['labels']):
            # Check the proximity between polylines
            d1, d2, check = polyline_merge_check(poly, poly2, proximity_th)
            # Ensure polylines have different labels or at least one polyline has label=0
            if label != label2 and (label ==0 or label2 ==0):
                check='False'
            # If proximity check passes, add an edge to the graph
            if check=='True':
                G.add_edge(idx, idx2)
   
    # Identify connected components in the graph
    C = [list(c) for c in nx.connected_components(G)]

    # Merge polylines within each connected component
    for idxs2merge in C:
        merged_poly = token.loc[idxs2merge[0]]
        for idx2 in idxs2merge[1:]:
            # Merge polylines using merge_polys_v1 function
            merged_poly['vectors']= merge_polys_v1(merged_poly['vectors'] ,token.loc[idx2]['vectors'])
            # If merged polylines have different labels, assign label 2
            if merged_poly['labels'] != token.loc[idx2]['labels']:
                merged_poly['labels'] = 2
        # Reset scores to 0
        merged_poly['scores'] = 0
        # Drop to be merged polylines from the token and append the merged polyline
        token_refined = token_refined.drop(idxs2merge)
        token_refined = token_refined.append(merged_poly)
    
    return token_refined.reset_index()

def refine_token_v2(token, proximity_th):
    """
    Refines a token containing polylines by merging polylines that are in close proximity to each other.

    Parameters:
    - token: DataFrame containing vectors, scores, and labels of polylines.
    - proximity_th: Proximity threshold determining when to merge polylines.

    Returns:
    - DataFrame containing refined polylines after merging.

    This function refines a token containing polylines by merging polylines that are in close proximity to each other. 
    It first creates a graph to store the connected polylines (nodes) to be merged. Then, it iterates over each pair 
    of polylines and checks if they satisfy the proximity threshold condition for merging. If the condition is met, 
    an edge is added to the graph connecting the two polylines. After identifying connected components in the graph, 
    polylines within each component are merged together. Finally, the refined token containing merged polylines is returned.
    """
    token_refined = pd.DataFrame()
    token_refined['vectors']= token['vectors']
    token_refined['scores']= token['scores']
    token_refined['labels']= token['labels']
    
    import networkx as nx
    # Create graph to store the connected polylines (nodes) to merge them at the end
    G = nx.Graph()

    for idx, poly, label in zip(token.index, token['vectors'], token['labels']):
        poly2_list = token.drop(idx)
        for idx2, poly2, label2 in zip(poly2_list.index, poly2_list['vectors'], poly2_list['labels']):
            d1, d2, check = polyline_merge_check(poly, poly2, proximity_th)
            if label != label2 :#and (label ==0 or label2 ==0):
                check='False'
            if check=='True':
                G.add_edge(idx, idx2)
    # Extract connected components from the graph
    C = [list(c) for c in nx.connected_components(G)]

    # Merge polylines within each connected component
    for idxs2merge in C:
        merged_poly = token.loc[idxs2merge[0]]
        for idx2 in idxs2merge[1:]:
            merged_poly['vectors']= merge_polys_v3(merged_poly['vectors'] ,token.loc[idx2]['vectors'])
        merged_poly['scores'] = 0
        token_refined = token_refined.drop(idxs2merge)
        token_refined = token_refined.append(merged_poly)
    
    return token_refined.reset_index()

def refine_token_v3(token, proximity_th):
    """
    Refines a token containing polylines by merging polylines that are in close proximity to each other.

    Parameters:
    - token: DataFrame containing vectors, scores, and labels of polylines.
    - proximity_th: Proximity threshold determining when to merge polylines.

    Returns:
    - DataFrame containing refined polylines after merging.

    This function refines a token containing polylines by merging polylines that are in close proximity to each other. 
    It first creates a graph to store the connected polylines (nodes) to be merged. Then, it iterates over each pair 
    of polylines and checks if they satisfy the proximity threshold condition for merging. If the condition is met, 
    an edge is added to the graph connecting the two polylines. After identifying connected components in the graph, 
    polylines within each component are merged together.

    For polylines with label 0 (representing ped_crossings), the function applies a rasterization approach to 
    merge them. It converts the polylines into raster images, calculates the average image, and then converts it back 
    into a simplified polygon. For polylines with labels other than 0, standard polyline merging (using `merge_polys_v2`) 
    is applied. Finally, the refined token containing merged polylines is returned.
    """
    import shapely.geometry, shapely.ops
    from rasterio import Affine, features
    from math import floor, ceil, sqrt

    token_refined = pd.DataFrame()
    token_refined['vectors']= token['vectors']
    token_refined['scores']= token['scores']
    token_refined['labels']= token['labels']
    import networkx as nx
    # Create graph to store the connected polylines (nodes) to merge them at the end
    G = nx.Graph()

    for idx, poly, label in zip(token.index, token['vectors'], token['labels']):
        poly2_list = token.drop(idx)
        for idx2, poly2, label2 in zip(poly2_list.index, poly2_list['vectors'], poly2_list['labels']):
            d1, d2, check = polyline_merge_check(poly, poly2, proximity_th)
            if label != label2: #and (label ==0 or label2 ==0):
                check='False'
            if check=='True':
                G.add_edge(idx, idx2)
    # Extract connected components from the graph
    C = [list(c) for c in nx.connected_components(G)]
    print(C)

    # Merge polylines within each connected component
    for idxs2merge in C:
        merged_poly = token.loc[idxs2merge[0]]
        print("idxs2merge: ", idxs2merge, " label: ", merged_poly['labels'])

        if merged_poly['labels']==0:
            polygons = []
            for cross in token.loc[idxs2merge]['vectors']:
                polygon = shapely.geometry.Polygon([[p[0], p[1]] for p in cross]).convex_hull
                polygons.append(polygon)

            max_shape = shapely.ops.unary_union(polygons)
            minx, miny, maxx, maxy = max_shape.bounds
            dx = dy = 0.05  # grid resolution; this can be adjusted
            lenx = dx * (ceil(maxx / dx) - floor(minx / dx))
            leny = dy * (ceil(maxy / dy) - floor(miny / dy))

            nx = int(lenx / dx)
            ny = int(leny / dy)
            gt = Affine(
                dx, 0.0, dx * floor(minx / dx),
                0.0, -dy, dy * ceil(maxy / dy))
            pa = np.zeros((ny, nx), 'd')
            for s in polygons:
                r = features.rasterize([s], (ny, nx), transform=gt)
                pa[r > 0] += 1
            pa /= len(polygons)  # normalize values
            spa, sgt = gaussian_blur(pa, gt, 100)
            thresh = 0.5  # median
            pm = np.zeros(spa.shape, 'B')
            pm[spa > thresh] = 1

            poly_shapes = []
            for sh, val in features.shapes(pm, transform=sgt):
                if val == 1:
                    poly_shapes.append(shapely.geometry.shape(sh))
            if not any(poly_shapes):
                raise ValueError("could not find any shapes")
            avg_poly = shapely.ops.unary_union(poly_shapes)
            # Simplify the polygon
            simp_poly = avg_poly.simplify(sqrt(dx**2 + dy**2))
            min_rect = simp_poly.minimum_rotated_rectangle

            min_rect_vec = min_rect.exterior.xy
            merged_poly['vectors'] = np.zeros((len(min_rect_vec[0]),2))
            merged_poly['vectors'] [:,0] = min_rect_vec[0]
            merged_poly['vectors'] [:,1] = min_rect_vec[1]
            
        else:
            for idx2 in idxs2merge[1:]:
                merged_poly['vectors']= merge_polys_v3(merged_poly['vectors'] ,token.loc[idx2]['vectors'])
                if merged_poly['labels'] != token.loc[idx2]['labels']:
                    merged_poly['labels'] = 2
        merged_poly['scores'] = 0
        token_refined = token_refined.drop(idxs2merge)
        token_refined = token_refined.append(merged_poly)
    
    return token_refined.reset_index()

#------------------------------------------------------------------------------------
# Frame transformation functions
def ego2world(e2g, egoPoints):
    """
    Function that transforms points from the ego frame to the world frame.

    Inputs:
    - e2g: List containing the translation vector and quaternion representing the transformation from ego to world frame.
      Format: [[tx, ty, tz], [qw, qx, qy, qz]]
        - tx, ty, tz: Translation along x, y, and z axes respectively.
        - qw, qx, qy, qz: Quaternion components representing rotation.

    - egoPoints: NumPy array of shape (2, N) representing points in the ego frame.
      Format: np.array([[x1, y1], [x2, y2], ..., [xm, ym]])

    Outputs:
    - worldPoints: NumPy array of shape (2, N) representing points in the world frame.
      Format: np.array([[x1', y1'], [x2', y2'], ..., [xm', ym']])
    """
    # Create quaternion object from the provided quaternion components
    quat = Quaternion(e2g[1])
    # Initialize array to store transformed points in world frame
    worldPoints = np.zeros((3,len(egoPoints[0])))
    # Iterate over each point in the ego frame
    for i in range(len(egoPoints[0])):
        # Apply rotation using the quaternion and add translation vector
        worldPoints[:,i] = quat.rotate(egoPoints[:,i])
        worldPoints[:,i] = worldPoints[:,i]+ np.array(e2g[0]).T 
    return worldPoints

def world2ego(e2g, worldPoints):
    """
    Function that transforms points from the world frame to the ego frame.

    Inputs:
    - e2g: List containing the translation vector and quaternion representing the transformation from ego to world frame.
      Format: [[tx, ty, tz], [qw, qx, qy, qz]]
        - tx, ty, tz: Translation along x, y, and z axes respectively.
        - qw, qx, qy, qz: Quaternion components representing rotation.

    - worldPoints: NumPy array of shape (2, N) representing points in the world frame.
      Format: np.array([[x1, y1], [x2, y2], ..., [xm, ym]])

    Outputs:
    - egoPoints: NumPy array of shape (2, N) representing points in the ego frame.
      Format: np.array([[x1', y1'], [x2', y2'], ..., [xm', ym']])
    """
    # Create inverse quaternion object from the provided quaternion components
    quat_inv = Quaternion(e2g[1]).inverse
    # Initialize array to store transformed points in ego frame
    egoPoints = np.zeros((3,len(worldPoints[0])))
    # Iterate over each point in the world frame
    for i in range(len(egoPoints[0])):
        # Subtract translation vector and apply inverse rotation using the inverse quaternion
        egoPoints[:,i] = worldPoints[:,i] - np.array(e2g[0]).T 
        egoPoints[:,i] = quat_inv.rotate(egoPoints[:,i])
    return egoPoints    

def token2rw(token, token_id, ego_TM):
    """
    Function that transforms a whole map token to the world frame.

    Inputs:
    - token: DataFrame containing map token(s) in ego frame.
    - token_id: Identifier for the token being transformed.
    - ego_TM: List of ego-to-world transformation matrices for all tokens.

    Outputs:
    - token: Transformed map token in the world frame.
    """
    # Iterate over each token in the DataFrame
    for i in token.index:
        # Extract the points of the current token and initialize array to store ego frame points
        egoPoints = np.zeros((3,len(token['vectors'][i])))
        egoPoints[:2,:] = token['vectors'][i].T # Transpose to get points in columns
        # Transform points from ego frame to world frame using the corresponding transformation matrix
        worldPoints = ego2world(ego_TM[token_id-5], egoPoints)
        # Update the token DataFrame with the transformed points
        token['vectors'][i][:,0] = worldPoints[0] # Update x coordinates
        token['vectors'][i][:,1] = worldPoints[1] # Update y coordinates
    return token


#------------------------------------------------------------------------------------
# Plot functions
def plotCar_quat(e2g, start_flag=False):
    """
    Function that plots a rotated rectangle representing a car.

    Inputs:
    - e2g: Transformation matrix representing ego-to-world translation and quaternion rotation.
    - start_flag: Boolean indicating whether it's the start position of the car. Default is False.
    """
    
    # Center point of the rectangle in ego frame
    cp_ego = np.array([0, 0, 0, 1])

    # Rectangle dimensions
    width = 6
    height = 3

    # Calculate the corner points of the rectangle
    top_left = (cp_ego[0] - width/2), (cp_ego[1] - height/2)
    top_right = (cp_ego[0] + width/2), (cp_ego[1] - height/2)
    bottom_left = (cp_ego[0] - width/2, cp_ego[1] + height/2)
    bottom_right = (cp_ego[0] + width/2, cp_ego[1] + height/2)
    corners = np.array([top_left, top_right, bottom_right, bottom_left])
    
    """
    # Translate then rotate the center and corners
    corners_ego = np.zeros((3,4))
    corners_ego[:2,:] = corners.T

    quat = Quaternion(e2g[1])
    corners_w = corners_ego
    for i in range(4):
        corners_w[:,i] = corners_ego[:,i]+ np.array(e2g[0]).T
        corners_w[:,i] = quat.rotate(corners_w[:,i])
    
    cp_w = cp_ego[:3] + e2g[0]
    cp_w = quat.rotate(cp_w)
    """
    # Rotate the center and corners from ego to world frame
    corners_ego = np.zeros((3,4))
    corners_ego[:2,:] = corners.T

    quat = Quaternion(e2g[1])
    corners_w = corners_ego
    for i in range(4):
        corners_w[:,i] = quat.rotate(corners_w[:,i])
        corners_w[:,i] = corners_ego[:,i]+ np.array(e2g[0]).T
    cp_w = quat.rotate(cp_ego[:3])
    cp_w = cp_w + e2g[0]

    # Plot the car rectangle
    plt.fill(corners_w[0], corners_w[1], fill=True, color='m', edgecolor='k')
    # Plot start position if start_flag is True
    if start_flag:
        plt.plot(cp_w[0], cp_w[1], marker='h', color='b', label='start_pos')
        plt.legend() 

def plotCarMap_quat(GlobalMap, e2g, token_id, threshold=0.8, start_flag=False):
    """
    Function that plots the full map token plus a rotated rectangle showing the car.

    Inputs:
    - GlobalMap: List of all map tokens.
    - e2g: Transformation matrix representing ego-to-world translation and quaternion rotation.
    - token_id: ID of the token to plot from the GlobalMap.
    - threshold: VectorMapNet prediction score threshold. Default is 0.8.
    - start_flag: Boolean indicating whether it's the start position of the car. Default is False.
    """
    # Center point of the rectangle in ego frame
    cp_ego = np.array([0, 0, 0, 1])

    # Rectangle dimensions
    width = 6
    height = 3

    # Calculate the corner points of the rectangle
    top_left = (cp_ego[0] - width/2), (cp_ego[1] - height/2)
    top_right = (cp_ego[0] + width/2), (cp_ego[1] - height/2)
    bottom_left = (cp_ego[0] - width/2, cp_ego[1] + height/2)
    bottom_right = (cp_ego[0] + width/2, cp_ego[1] + height/2)
    corners = np.array([top_left, top_right, bottom_right, bottom_left])

    # Rotate then translate the center and corners
    corners_ego = np.zeros((3,4))
    corners_ego[:2,:] = corners.T

    quat = Quaternion(e2g[1])
    corners_w = corners_ego
    for i in range(4):
        corners_w[:,i] = quat.rotate(corners_w[:,i])
        corners_w[:,i] = corners_ego[:,i]+ np.array(e2g[0]).T
    
    
    cp_w = quat.rotate(cp_ego[:3])
    cp_w = cp_w + e2g[0]

    # Rotate then translate the polylines
    token_ex = GlobalMap.iloc[token_id, :].results
    for i in range(len(token_ex['vectors'])):
        if token_ex['scores'][i] >= threshold:
            x_values = token_ex['vectors'][i][:,0]
            y_values = token_ex['vectors'][i][:,1]
            label = token_ex['labels'][i]
            xy_values = np.zeros((3,len(x_values)))
            xy_values[0,:]= x_values
            xy_values[1,:]= y_values

            for i in range(len(x_values)):
                xy_values[:,i] = quat.rotate(xy_values[:,i])
                xy_values[:,i] = xy_values[:,i]+ np.array(e2g[0]).T            

            if label == 0:
                c = 'g'
                label = 'ped_crossing'
            elif label == 1:
                c = 'r'
                label = 'divider'
            else: 
                c = 'k'
                label = 'boundary'
            plt.plot(xy_values[0,:], xy_values[1,:], color=c, label=label)
            
        else: pass

    
    plt.fill(corners_w[0], corners_w[1], fill=True, color='m', edgecolor='k');
    
    if start_flag:
        plt.plot(cp_w[0], cp_w[1], marker='h', color='b', label='start_pos');
        
    green = Line2D([0],[0],color='g', lw=1, label='ped_cross')
    red = Line2D([0],[0],color='r', lw=1, label='divider')
    black = Line2D([0],[0],color='k', lw=1, label='boundary')
    magenta = patches.Patch(facecolor='m',edgecolor='k', label='car')
    blue = Line2D([0],[0], marker='h',color='w', markerfacecolor='b',markersize=9, label='start_pos')
    plt.legend(handles=[green, red, black, magenta, blue])
    plt.axis('square')
    plt.xlabel('Position in X direction (m)')
    plt.ylabel('Position in Y direction (m)')
    
def plot_poly(poly, name="poly", color='r', square=True):
    """
    Function that plots a single polyline.

    Inputs:
    - poly: Array of polyline points.
    - name: Name of the polyline for the legend. Default is "poly".
    - color: Color of the polyline. Default is 'r' (red).
    - square: Flag to set the figure square or not. Default is True.
    """
    # Plot the polyline
    plt.plot(poly[:,0], poly[:,1], label=name, color=color)
    # Plot a marker at the start point
    plt.plot(poly[0,0], poly[0,1], 'o', label='start', color=color)
    # Set plot title and labels
    plt.title("Polyline")
    plt.xlabel('Position in X direction (m)')
    plt.ylabel('Position in Y direction (m)')
    # Add legend
    plt.legend()
    # Set figure aspect ratio to square if square=True
    if square==True:    
        plt.axis('square')

def plot_merge_2polys(poly1, poly2, square=True, v=2):
    """
    Function that merges two polys then plots them.

    Inputs:
    - poly1: First polyline.
    - poly2: Second polyline.
    - square: Flag to set the figure square or not. Default is True.
    - v: Version of the merge_polys function to use. Default is 2.
    """
    # Merge the two polylines based on the specified version
    if v==0:
        merged_poly = merge_polys_v1(poly1,poly2)
    elif v==1:
        merged_poly = merge_polys_v2(poly1,poly2)
    elif v==2:
        merged_poly = merge_polys_v3(poly1,poly2)
        
    # Plot the original polylines
    plt.plot(poly1[:,0], poly1[:,1], c='k', label='Old')
    plt.plot(poly2[:,0], poly2[:,1], c='g', label='New')
    # Plot the merged polyline
    plt.plot(merged_poly[:,0], merged_poly[:,1], label='Merged', color='m')
    # Plot markers at the starting points of the original polylines and the merged polyline
    plt.plot(merged_poly[0,0], merged_poly[0,1], 'mo', label='start_M')
    plt.plot(poly1[0,0], poly1[0,1], 'ko')#, label='start_p1')
    plt.plot(poly2[0,0], poly2[0,1], 'go')#, label='start_p2')
    # Set plot title and labels
    plt.title("Merged poly")
    plt.xlabel('Position in X direction (m)')
    plt.ylabel('Position in Y direction (m)')
    # Add legend
    plt.legend()
    # Set plot limits based on the maximum and minimum coordinates of the original and merged polylines
    x_left = min([min(poly1[:,0]), min(poly2[:,0]), min(merged_poly[:,0])])-1
    x_right = max([max(poly1[:,0]), max(poly2[:,0]), max(merged_poly[:,0])])+1
    y_down = min([min(poly1[:,1]), min(poly2[:,1]), min(merged_poly[:,1])])-1
    y_up = max([max(poly1[:,1]), max(poly2[:,1]), max(merged_poly[:,1])])+1
    plt.xlim((x_left, x_right))
    plt.ylim((y_down, y_up))
    # Set figure aspect ratio to square if square=True
    if square==True:    
        plt.axis('square')

def plot_token(token, name="token"):
    """
    Function to plot a full map token.

    Inputs:
    - token: DataFrame containing map token information, including polyline vectors and labels.
    - name: Name of the token for the legend. Default is "token".
    """
    for i in token.index:
        poly = token.loc[i]

        x_values = poly['vectors'][:,0]
        y_values = poly['vectors'][:,1]
        if poly['labels'] == 0:
            color = 'g'
            label = 'ped_crossing'
        elif poly['labels'] == 1:
            color = 'r'
            label = 'divider'
        else: 
            color = 'k'
            label = 'boundary'
        plt.plot(x_values, y_values, color=color)#, label=i)
        plt.axis('square')
        
    # Create legend for different labels
    green = Line2D([0],[0],color='g', lw=1, label='ped_cross')
    red = Line2D([0],[0],color='r', lw=1, label='divider')
    black = Line2D([0],[0],color='k', lw=1, label='boundary')
    magenta = patches.Patch(facecolor='m',edgecolor='k', label='car')
    blue = Line2D([0],[0], marker='h',color='w', markerfacecolor='b',markersize=9, label='start_pos')
    # Add legend to the plot
    plt.legend(handles=[green, red, black, magenta, blue])
    # Set aspect ratio to square
    plt.axis('square')
    
#------------------------------------------------------------------------------------
# Extra functions
def delete_duplicate(poly):
    """
    Function that removes duplicate points from a polyline.

    Inputs:
    - poly: numpy array representing the polyline with duplicate points.

    Outputs:
    - poly_unique: numpy array representing the polyline with duplicate points removed.
    """
    # Initialize an array to store unique points
    poly_unique = np.array([[poly[0,0], poly[0,1]]])
    # Iterate over each point in the polyline
    for point in poly:
        # Check if the point is already in the unique array
        if point in poly_unique:
            pass  # Skip if the point is a duplicate
        else:
            # Add the point to the unique array if it's not a duplicate
            new_point = np.array([[point[0], point[1]]])
            poly_unique = np.append(poly_unique, new_point, axis=0)
    return poly_unique

def smooth_poly(poly, smooth_th):
    """
    Function that smooths a polyline by removing points that are close to each other.

    Inputs:
    - poly: numpy array representing the polyline to be smoothed.
    - smooth_th: threshold distance for smoothing. Points closer than this distance will be removed.

    Outputs:
    - polySmooth: numpy array representing the smoothed polyline.
    """
    # Initialize the smoothed polyline as the original polyline
    polySmooth = poly
    # Iterate over each pair of consecutive points in the polyline
    for p, p_ in zip(polySmooth[0:-1],polySmooth[1:]):
        # Calculate the distance between two consecutive points
        if math.dist(p, p_) < smooth_th:
            # If the distance is less than the threshold, calculate the midpoint
            pm = (p+p_)/2
            # Find the index of the first point in the pair
            idx = np.argwhere(np.all(polySmooth == p, axis=1))[0][0]
            # Delete both points from the polyline
            polySmooth = np.delete( polySmooth, [idx, idx+1], axis=0 )
            # Insert the midpoint at the index of the first point
            polySmooth = np.insert(polySmooth, idx, pm, axis=0)
            
    return polySmooth

def insertPoint(p,p2,p3,p4, polyM):
    """
    Function that inserts a point into a polyline based on the relative positions of surrounding points.

    Inputs:
    - p: The point to be inserted.
    - p2, p3: Two consecutive points in the polyline.
    - p4: projection of p on p2-p3
    - polyM: Numpy array representing the polyline.

    Outputs:
    - polyM: Numpy array representing the updated polyline after inserting the point.
    - direction: A string indicating the direction of insertion ('before', 'after', or 'in').
    """    
    # Initialize direction as NaN
    direction = np.nan
    # Calculate unit vectors for p2p3, p2p4, and p3p4
    p2p3_u = (p3-p2) / np.linalg.norm(p3-p2)
    p2p4_u = (p4-p2) / np.linalg.norm(p4-p2) 
    p3p4_u = (p4-p3) / np.linalg.norm(p4-p3)

    # Find the indices of p2 and p3 in polyM
    p2_index = np.argwhere(np.all(polyM == p2, axis=1))[0][0]
    p3_index = np.argwhere(np.all(polyM == p3, axis=1))[0][0]
    
    # Check the scenarios: s1 (p before p2), s2 (p after of p3), s3 (p is in p2-p3)
    if all(np.sign(p2p3_u) == np.sign(-p2p4_u)):
        direction = "before"
        polyM = np.insert(polyM, p2_index, p, axis=0)

    elif all(np.sign(p2p4_u) == np.sign(p3p4_u)):
        direction = "after"
        polyM = np.insert(polyM, p3_index+1, p, axis=0)

    else:
        direction = "in"
        polyM = np.insert(polyM, p2_index+1,p, axis=0)
    return polyM, direction
        
def gaussian_blur(in_array, gt, size):
    """
    Applies Gaussian blur to the input array.

    Inputs:
    - in_array: Input array to be blurred.
    - gt: Geotransform of the input array, specifying its spatial properties.
    - size: Size of the Gaussian kernel. Larger sizes result in stronger blur effects.

    Outputs:
    - ar: The blurred array after convolution with the Gaussian kernel.
    - gt2: The updated geotransform reflecting the change in size of the output array.

    Gaussian blur is a common image processing technique used for noise reduction and smoothing. 
    This function first pads the input array with zeros to accommodate the edge effects of convolution. Then, it constructs
    a Gaussian kernel based on the specified size. The kernel is normalized to ensure that the sum of all elements equals
    1. Next, the input array is convolved with the Gaussian kernel using the 'full' convolution mode, which ensures that
    the output array has the correct size. Finally, the geotransform is updated to reflect the change in size of the array,
    and the blurred array along with the updated geotransform is returned.
    """
    from scipy.signal import fftconvolve
    from rasterio import Affine, features
    
    # Expand in_array to fit edge of kernel; constant value is zero
    padded_array = np.pad(in_array, size, 'constant')
    
    # Build Gaussian kernel
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
    g = (g / g.sum()).astype(in_array.dtype)
    
    # Perform Gaussian blur
    ar = fftconvolve(padded_array, g, mode='full')
    
    # Update geotransform
    gt2 = Affine(
        gt.a, gt.b, gt.xoff - (2 * size * gt.a),
        gt.d, gt.e, gt.yoff - (2 * size * gt.e))
    return ar, gt2