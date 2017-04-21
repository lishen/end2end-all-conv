import numpy as np
from skimage.measure import label, regionprops


def region_features(region=None):
    if region is None:
        return {
            'area':0, 'area_ratio':0., 'area_ratio2':0., 'eccentricity':0.,
            'equivalent_diameter':0., 'euler_number':2, 'extent':0.,
            'eig1':0., 'eig2':0., 'major_axis_length':0., 'max_intensity':0.,
            'mean_intensity':0., 'minor_axis_length':0., 'orientation':0.,
            'perimeter':0., 'solidity':0.,
        }
    return {
        'area':region.area,
        'area_ratio':float(region.area)/region.convex_area,
        'area_ratio2':float(region.area)/region.filled_area,
        'eccentricity':region.eccentricity,
        'equivalent_diameter':region.equivalent_diameter,
        'euler_number':region.euler_number,
        'extent':region.extent,
        'eig1':region.inertia_tensor_eigvals[0],
        'eig2':region.inertia_tensor_eigvals[1],
        'major_axis_length':region.major_axis_length,
        'max_intensity':region.max_intensity,
        'mean_intensity':region.mean_intensity,
        'minor_axis_length':region.minor_axis_length,
        'orientation':region.orientation,
        'perimeter':region.perimeter,
        'solidity':region.solidity,
    }

def total_area(regions=[]):
    areas = [ reg.area for reg in regions]
    return sum(areas)

def global_max_intensity(regions=[]):
    max_int = [ reg.max_intensity for reg in regions]
    return max(max_int) if len(max_int) > 0 else 0.0

def topK_region_idx(regions, k=1):
    areas = [ reg.area for reg in regions]
    return np.argsort(areas)[-1:-(k+1):-1]

def prob_heatmap_features(phm, cutoff, k=1, nb_cls=3):
    fea_list = []
    if phm is None:  # deal with missing view.
        for _ in xrange(nb_cls - 1): # phms depending on the # of cls.
            fea = {'nb_regions': np.nan, 'total_area': np.nan, 
                   'global_max_intensity': np.nan}
            for j in xrange(k):
                reg_fea = {
                    'area': np.nan, 'area_ratio': np.nan, 'area_ratio2': np.nan,
                    'eccentricity': np.nan, 'eig1': np.nan, 'eig2': np.nan,
                    'equivalent_diameter': np.nan, 'euler_number': np.nan, 
                    'extent': np.nan, 
                    'major_axis_length': np.nan, 'max_intensity': np.nan,
                    'mean_intensity': np.nan, 'minor_axis_length': np.nan,
                    'orientation': np.nan, 'perimeter': np.nan,
                    'solidity': np.nan, 
                }
                for key in reg_fea.keys():
                    new_key = 'top' + str(j+1) + '_' + key
                    reg_fea[new_key] = reg_fea.pop(key)
                fea.update(reg_fea)
            fea_list.append(fea)
        return fea_list
    
    for i in xrange(1, nb_cls):
        phm_ = phm[:,:,i]
        hm_bin = np.zeros_like(phm_, dtype='uint8')
        hm_bin[phm_ >= cutoff] = 255
        hm_label = label(hm_bin)
        props = regionprops(hm_label, phm_)
        fea = {
            'nb_regions':len(props),
            'total_area':total_area(props),
            'global_max_intensity':global_max_intensity(props),
        }
        nb_reg = min(k, len(props))
        idx = topK_region_idx(props, k)
        for j,x in enumerate(idx):
            reg_fea = region_features(props[x])
            for key in reg_fea.keys():
                new_key = 'top' + str(j+1) + '_' + key
                reg_fea[new_key] = reg_fea.pop(key)
            fea.update(reg_fea)
        for j in xrange(nb_reg, k):
            reg_fea = region_features()
            for key in reg_fea.keys():
                new_key = 'top' + str(j+1) + '_' + key
                reg_fea[new_key] = reg_fea.pop(key)
            fea.update(reg_fea)
        fea_list.append(fea)
    return fea_list







