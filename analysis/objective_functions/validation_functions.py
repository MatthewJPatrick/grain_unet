#from file_io import *
from scipy.spatial import KDTree, distance
import numpy as np
try:
    from centroids import Centroid
except:
    pass

#This file contains the support functions for the U-Net validation scheme

#-------------------------------------------------------------------------------
# !!!! maybe wrong !!! This function takes one list of points, which may be identical or not,
# and for each point in the comparison list finds the nearest neighbor in the reference list
# Returns a tuple: (a list where each element contains ([[x,y], [x_nn,y_nn], distance]; the average NN distance) !!!maybe wrong!!!!
#-------------------------------------------------------------------------------
def find_neighbors(centroids):
    if isinstance(centroids[0], Centroid):
        points = [centroid.coords for centroid in centroids]
    else: 
        points = centroids

    neighbors = [[[0,0], [0,0], 0]]*len(points)
    total_dist = 0

    tree = KDTree(points)

    ii = 0
    for point in points:
        dist, idx = tree.query(point, k=2)
        
        neighbors[ii] = [centroids[ii], centroids[idx[1]], dist[1]]
        ii+=1
        total_dist += dist[1]

    mean_dist = total_dist/len(points)
    return {'neighbors' : neighbors, 'mean_dist' :mean_dist}
#----------------------------------------------------------------------
# This function takes in two lists of points comp_points and ref_points,
# and returns a list of pairs of points such that for each point in 
# comp_points, it is paired with the nearest point in ref_points within
# if their distance is less than some cutoff distance.
#----------------------------------------------------------------------

def find_pairs(comp_points, ref_points, cutoff = 0.3, mean_dist = 100.0, orphan_list = False):
    
    # if len(comp_points) < len(ref_points):
    #     max_pairs = len(comp_points)
    # else:
    #     max_pairs = len(ref_points)

    pairs = list()
    shortest_distances = list()
    for comp_point in comp_points:
        distances = [distance.euclidean(comp_point.coords, point.coords) for point in ref_points]
        nearest_idx = np.argmin(distances)

        # check if the comparison point is within the cutoff for its nearest point in the reference set
        if distances[nearest_idx] < cutoff*mean_dist:
            
            dist = distances[nearest_idx]
            ref_point = ref_points[nearest_idx]
            existing_ref_points = [element[1] for element in pairs]

            # check if the reference point already has been paired with a comparison point, i.e. it is 
            # already in the pairs set. If it is not, then proceed as normal and append the pair to pairs 
            # and the distance to distances.
            if not ref_point in existing_ref_points:
                pairs.append([comp_point,ref_point])
                shortest_distances.append(dist)
            
            # if it is in the set (else), find the index of the reference point in the existing reference point list.
            else:
                existing_ref_point_index = existing_ref_points.index(ref_point)
            
                # check if the distance between the new comparison point and reference point
                # is less than the distance in existing shortest distance list (distances). This element will have the same
                # index as the reference point in the existing_reference_points extracted from the pairs list.

                if dist < distances[existing_ref_point_index]:
                    pairs[existing_ref_point_index][1] = ref_point
                    shortest_distances[existing_ref_point_index] = dist

    if orphan_list: 
        pairs = np.array(pairs)
        comp_orphans = []
        ref_orphans = []
        for point in comp_points:
            if point not in pairs[:,0]:
                comp_orphans.append(point)
        for point in ref_points:
            if point not in pairs[:,1]: 
                ref_orphans.append(point)
        return comp_orphans, ref_orphans

    comp_orphans = len(comp_points)-len(pairs)
    ref_orphans = len(ref_points) - len(pairs)
    orphans = (comp_orphans + ref_orphans)
    return{'pairs' : pairs, 'distances' : shortest_distances, 'orphans' : orphans}

#-------------------------------------------------------------------------------
# This is a function which calculates the normalize disregistry from a list of 
# pairs of points
#-------------------------------------------------------------------------------

def disregistry(pairs, cutoff, mean_dist):
    worst_case = len(pairs)*cutoff*mean_dist
    disregistry = sum(pairs)/worst_case
    return disregistry

#-------------------------------------------------------------------------------
# This is a function to take a file name and extract the post processing parameters
#-------------------------------------------------------------------------------

def get_params(path):
    path_str = f'{path}'
    post_parameters ={
        'compilation': 'min',
        'liberal_thresh': path_str.split('lib')[1][1:4],     ##220-180 for best axon
        'conservative_thresh': path_str.split('cons')[1][1:4], 
        'invert_double_thresh': True,
        'n_dilations': path_str.split('dils')[1][1:4],
        'min_grain_area': path_str.split('minarea')[1][1:4], #usually 300, cut to 100 for 2019 data
        'prune_size': path_str.split('prune')[1][1:4], #usually 100
        'out_dict': True
        }  
    return post_parameters

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

#-------------------------------------------------------------------------------
# This is a function to compute the disregistry and orphan fraction for a pair
# of lists of centroids.
#-------------------------------------------------------------------------------

def get_objectives(ref_centroids, comp_centroids, cutoff_fraction=0.3):

    #find nearest neighbor distand in the reference image. Find pairs between ref and comparison data
    mean_nn_dist = find_neighbors(ref_centroids)['mean_dist']
    pairs = find_pairs(comp_centroids, ref_centroids, cutoff = cutoff_fraction, mean_dist = mean_nn_dist)

    #calculate normalized disregistry and orphan scores
    disreg = disregistry(pairs['distances'], 0.3, mean_nn_dist)
    orphan_frac = pairs['orphans']/(len(comp_centroids)+len(ref_centroids))


    return {'disregistry': disreg, 'orphans' : orphan_frac, 'n_pairs' : len(pairs['distances'])}

def get_orphans(ref_centroids, comp_centroids, cutoff_fraction = 0.3):
    #find nearest neighbor distand in the reference image. Find pairs between ref and comparison data
    mean_nn_dist = find_neighbors(ref_centroids)['mean_dist']
    comp_orphans, ref_orphans = find_pairs(comp_centroids, ref_centroids, cutoff = cutoff_fraction, mean_dist = mean_nn_dist, orphan_list = True)
    return ref_orphans, comp_orphans
