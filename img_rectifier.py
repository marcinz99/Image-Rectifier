import sys
import numpy as np
from PIL import Image
from time import time

SPLITTING_THRESHOLD = 0 # Globals to be initialized in rectifier()
MINIMUM_BOX_SIZE = 0    #


def row_mean_cumsum(array):
    return np.cumsum(np.sum(array, axis=0) / array.shape[1])

def col_mean_cumsum(array):
    return np.cumsum(np.sum(array, axis=1) / array.shape[0])

def quick_median(array):
    bins, counts = np.unique(array, return_counts=True)
    pixel_num = array.shape[0] * array.shape[1]
    threshold = pixel_num // 2
    counts = np.cumsum(counts)
    
    for i, val in enumerate(bins):
        if counts[i] > threshold:
            return val


def find_split_line(array, vert=True):
    if vert:
        cols_cs = col_mean_cumsum(array)
        max_len = cols_cs.shape[0]
        sum_all = cols_cs[-1]
        
        best_split = 0
        best_score = 1.0
        
        for i in range(0, max_len-1):
            left  = cols_cs[i] / (i + 1)
            right = (sum_all - cols_cs[i]) / (max_len - i - 1)
            
            if left / right > best_score:
                best_score = left / right
                best_split = i
            elif right / left > best_score:
                best_score = right / left
                best_split = i
        
        return best_split
    else:
        rows_cs = row_mean_cumsum(array)
        max_len = rows_cs.shape[0]
        sum_all = rows_cs[-1]
        
        best_split = 0
        best_score = 1.0
        
        for i in range(0, max_len-1):
            left  = rows_cs[i] / (i + 1)
            right = (sum_all - rows_cs[i]) / (max_len - i - 1)
            
            if left / right > best_score:
                best_score = left / right
                best_split = i
            elif right / left > best_score:
                best_score = right / left
                best_split = i
        
        return best_split


def refine(array, depth=30, x_from=-1, x_to=-1, y_from=-1, y_to=-1, vert=True, end_handler=None):
    # Box splitter. Applies a transformer on the final boxes.
    global SPLITTING_THRESHOLD, MINIMUM_BOX_SIZE
    
    if x_from == -1 or x_to == -1 or y_from == -1 or y_to == -1:
        split = find_split_line(array, vert=True)
        x_from, x_to = 0, array.shape[0]
        y_from, y_to = 0, array.shape[1]
        
        refine(array, depth-1, x_from, x_from+split+1, y_from, y_to, False, end_handler)
        refine(array, depth-1, x_from+split+1, x_to, y_from, y_to, False, end_handler)
    
    elif depth == 0 or (x_to-x_from==1 and vert) or (y_to-y_from==1 and not(vert))\
            or (x_to-x_from)*(y_to-y_from)<SPLITTING_THRESHOLD:
        if not end_handler:
            array[x_from:x_to, y_from:y_to] = np.random.randint(0, 256, dtype=np.uint8)
        else:
            end_handler(array, x_from, x_to, y_from, y_to)
        
    else:
        if vert:
            split = find_split_line(array[x_from:x_to, y_from:y_to], vert=True)
            split = max(split, int(0.15*(x_to-x_from)))
            split = min(int(0.85*(x_to-x_from)), split)
            
            while (split+1)*(y_to-y_from) < MINIMUM_BOX_SIZE:
                split += 1
            while (x_to-x_from-split-1)*(y_to-y_from) < MINIMUM_BOX_SIZE:
                split -= 1

            refine(array, depth-1, x_from, x_from+split+1, y_from, y_to,
                   split+1>y_to-y_from, end_handler)
            refine(array, depth-1, x_from+split+1, x_to, y_from, y_to,
                   x_to-x_from-split-1>y_to-y_from, end_handler)
        else:
            split = find_split_line(array[x_from:x_to, y_from:y_to], vert=False)
            split = max(split, int(0.15*(y_to-y_from)))
            split = min(int(0.85*(y_to-y_from)), split)
            
            while (x_to-x_from)*(split+1) < MINIMUM_BOX_SIZE:
                split += 1
            while (x_to-x_from)*(y_to-y_from-split-1) < MINIMUM_BOX_SIZE:
                split -= 1

            refine(array, depth-1, x_from, x_to, y_from, y_from+split+1,
                   x_to-x_from>split+1, end_handler)
            refine(array, depth-1, x_from, x_to, y_from+split+1, y_to,
                   x_to-x_from>y_to-y_from-split-1, end_handler)


# === TRANSFORMERS ===============================================================
# You can define your own transformers here, which will be used on the boxes

def median_transform_basic(array, x_from, x_to, y_from, y_to):
    median = quick_median(array[x_from:x_to, y_from:y_to])
    scale = 255. / median
    array[x_from:x_to, y_from:y_to] =\
        np.minimum(scale*array[x_from:x_to, y_from:y_to], 255).astype(np.uint8)
    
def median_transform_high_contrast(array, x_from, x_to, y_from, y_to):
    # Recommended transformer
    median = quick_median(array[x_from:x_to, y_from:y_to])
    scale = 1.1 / median
    
    array[x_from:x_to, y_from:y_to] =\
        np.minimum(255*np.power(scale*array[x_from:x_to, y_from:y_to], 4), 255).astype(np.uint8)

transformers = {
    'median_contrast': median_transform_high_contrast,
    'median_basic':    median_transform_basic,
    'show_boxes':      None # Option for devs wanting to see the boxes
}
# ================================================================================


def rectifier(filename, transformer, maxdepth=50, split_factor=200):
    # filepath - path to file
    # transformer - end-of-recursion handler for refine(), transforms boxes
    # maxdepth - maximum depth of tree divisions of the image
    # split_factor - SPLITTING THRESHOLD is defined as size_in_pixels/split_factor
    global SPLITTING_THRESHOLD, MINIMUM_BOX_SIZE
    
    img = Image.open(filename)
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    
    size_in_pixels = img_array.shape[0] * img_array.shape[1]
    SPLITTING_THRESHOLD = int(size_in_pixels/split_factor)
    MINIMUM_BOX_SIZE = int(0.4 * SPLITTING_THRESHOLD)
    
    refine(img_array, depth=maxdepth, end_handler=transformer)
    return img_array.astype(np.uint8)
    

# === FURTHER PROCESSING =========================================================
# Additional cleansing steps to apply to the image

def entropy_reduction(img_array, decay_range, edge_fuzz, entropy_cutoff,
        white_cutoff, threshold_quantile, cleanse_strength):
    # This tool finds regions of high local (pseudo-)entropy and whitens them.
    center = img_array[1:-1, 1:-1]
    decay = np.power(0.01, 1./decay_range)

    vicinity_variability = np.zeros(center.shape)
    vertical_entropy = np.zeros(center.shape)
    horizontal_entropy = np.zeros(center.shape)
    local_entropy = np.zeros(center.shape)

    x_high, y_high = img_array.shape
    x_high, y_high = x_high - 1, y_high - 1

    for xi, yi in [( 1, -1), ( 1,  0), ( 1,  1), ( 0, -1),
                   ( 0,  1), (-1, -1), (-1,  0), (-1,  1)]:
        vicinity_variability += np.abs(center - img_array[1-xi : x_high-xi, 1-yi : y_high-yi])

    width  = vicinity_variability.shape[1]
    height = vicinity_variability.shape[0]
    
    aggr = np.zeros(width)
    for i in range(height):
        aggr = decay * aggr + vicinity_variability[i, :]
        vertical_entropy[i, :] += aggr

    aggr = np.zeros(width)
    for i in range(height-1, -1, -1):
        aggr = decay * aggr + vicinity_variability[i, :]
        vertical_entropy[i, :] += aggr

    aggr = np.zeros(height)
    for i in range(width):
        aggr = decay * aggr + vertical_entropy[:, i]
        local_entropy[:, i] += aggr

    aggr = np.zeros(height)
    for i in range(width-1, -1, -1):
        aggr = decay * aggr + vertical_entropy[:, i]
        local_entropy[:, i] += aggr
    
    base = 20 * edge_fuzz * 8 * 255 / (1 - decay)
    factor = 1.0
    i = 0
    while factor > 0.02:
        val = factor * base
        if i == 0:
            local_entropy[: , 0 ] += val
            local_entropy[: , -1] += val
            local_entropy[0 , 1:-1] += val
            local_entropy[-1, 1:-1] += val
        else:
            local_entropy[i:-i, i   ] += val
            local_entropy[i:-i, -1-i] += val
            local_entropy[i   , i+1:-i-1] += val
            local_entropy[-1-i, i+1:-i-1] += val
        
        factor *= decay
        i += 1
    
    entropy_map = np.zeros(img_array.shape)
    entropy_map[1:-1, 1:-1] = local_entropy
    entropy_map[:, 0] = entropy_map[:, 1]
    entropy_map[:, -1] = entropy_map[:, -2]
    entropy_map[0, :] = entropy_map[1, :]
    entropy_map[-1, :] = entropy_map[-2, :]
    
    max_possible = 8 * 255 / (1 - decay)**2
    entropy_adjustment = np.maximum(0, (entropy_map - entropy_cutoff * max_possible))
    entropy_adjustment /= np.max(entropy_adjustment)

    result = np.minimum(255, (img_array + cleanse_strength * 255 * entropy_adjustment)).astype(np.uint8)

    if threshold_quantile > 0:
        probes = np.zeros(2000)
        for i in range(2000):
            x, y = np.random.randint(height), np.random.randint(width)
            probes[i] = local_entropy[x, y]

        threshold = np.quantile(probes, threshold_quantile)
        result = np.where(np.greater(entropy_map, threshold), result, 255)

    result = np.where(np.less(result, white_cutoff), result, 255)
    return result.astype(np.uint8)

def disjoint_sets_optimization(img_array, cluster_threshold):
    # Suppose that adjacent non-white pixels form a cluster and each cluster
    # has an aggregate value of total blackness (255 - regular grey-scale value)
    # of the pixels in that cluster.
    # This tool estimates the lower bound for its cluster value and whitenes
    # those below the threshold.
    clusters_f = (255 - img_array).astype(np.int32)
    clusters_b = clusters_f.copy()

    for i in range(1, img_array.shape[0]):
        clusters_f[i, :] += np.where(np.greater(clusters_f[i, :], 0), clusters_f[i-1, :], 0)

    for j in range(1, img_array.shape[1]):
        clusters_f[:, j] += np.where(np.greater(clusters_f[:, j], 0), clusters_f[:, j-1], 0)

    for i in reversed(range(0, img_array.shape[0]-1)):
        clusters_b[i, :] += np.where(np.greater(clusters_b[i, :], 0), clusters_b[i+1, :], 0)

    for j in reversed(range(0, img_array.shape[1]-1)):
        clusters_b[:, j] += np.where(np.greater(clusters_b[:, j], 0), clusters_b[:, j+1], 0)

    clusters = clusters_f + clusters_b
    result = np.where(np.greater(clusters, cluster_threshold), img_array, 255)
    return result.astype(np.uint8)

# ================================================================================

def help():
    print(" ==== H E L P   S E C T I O N ==== ")
    print("Major options:")
    print("  -e      - Add entropy reduction to pipeline")
    print("  -d      - Add disjoint sets optimization to pipeline")
    print("  -i      - Inplace")
    print("  -p=     - Prefix")
    print("Parameter-setting options:")
    print("  Box transformation:")
    print("  -shbox  - Show-boxes transformer")
    print("  -medba  - Basic median transformer")
    print("  -medco  - Median transformer with high contrast (default)")
    print("  -maxde= - Max depth")
    print("  -split= - Split factor")
    print("  Entropy reduction:")
    print("  -decay= - Decay range")
    print("  -edgef= - Edge fuzz")
    print("  -entcu= - Entropy cutoff")
    print("  -whicu= - White cutoff")
    print("  -thrqu= - Threshold quantile")
    print("  -clstr= - Cleanse strength")
    print("  Disjoint sets optimization:")
    print("  -clust= - Cluster threshold")
    print(" ==== E N D ==== ")

if __name__ == '__main__':
    arg_n = len(sys.argv)    
    filename = None
    pipeline = []
    
    # === SETUP ===========================================================
    # Feel free to hardcode the desirable options as deafult parameters
    
    # General options:
    inplace = False # If true, enhanced photo overwrites the existing one
    prefix = "_"
    
    # Box transformation parameters:
    transformer = transformers['median_contrast']
    split_factor = 1000
    maxdepth = 150
    
    # Entropy reduction parameters:
    decay_range = 100    
    edge_fuzz = 0.5
    entropy_cutoff = 0.6
    white_cutoff = 192
    threshold_quantile = 0.15
    cleanse_strength = 25
    
    # Disjoint sets optimization parameters:
    cluster_threshold = 2000
    
    # =====================================================================
    
    for i in range(1, arg_n):
        str_i = str(sys.argv[i])
        if i == 1 and str_i == '?':
            help()
            break
        elif str_i[0] == '-':
            if len(str_i) == 2:
                if   str_i[1] == 'e': pipeline += ['e']
                elif str_i[1] == 'd': pipeline += ['d']
                elif str_i[1] == 'i': inplace = True
            elif len(str_i) >= 3 and str_i[1:3] == 'p=': prefix = str_i[3:]
            elif len(str_i) == 6:
                if   str_i[1:6] == 'shbox': transformer = transformers['show_boxes']
                elif str_i[1:6] == 'medba': transformer = transformers['median_basic']
                elif str_i[1:6] == 'medco': transformer = transformers['median_contrast']
            elif len(str_i) >= 7:
                param = str_i[7:]
                if not param.replace('.', '', 1).isdigit():
                    print("Invalid parameter value:", str_i)
                elif str_i[1:7] == 'maxde=': maxdepth = int(param)
                elif str_i[1:7] == 'split=': split_factor = int(param)
                elif str_i[1:7] == 'decay=': decay_range = int(param)
                elif str_i[1:7] == 'edgef=': edge_fuzz = float(param)
                elif str_i[1:7] == 'entcu=': entropy_cutoff = float(param)
                elif str_i[1:7] == 'whicu=': white_cutoff = int(param)
                elif str_i[1:7] == 'thrqu=': threshold_quantile = float(param)
                elif str_i[1:7] == 'clstr=': cleanse_strength = float(param)
                elif str_i[1:7] == 'clust=': cluster_threshold = int(param)
            else:
                print("Invalid option:", str_i)
        else:
            filename = str_i
            
    if arg_n <= 1:
        print("Nothing to do here. Type argument '?' to see help section.")
    elif not filename:
        print("Specify the filename (remember not to pass filepath!).")
    
    else:
        start_time = time()
        rectified = rectifier(filename, transformer, maxdepth, split_factor)
        new_name = filename if inplace else prefix + filename
        
        for step in pipeline:
            if step == 'e':
                rectified = entropy_reduction(
                    rectified, decay_range, edge_fuzz, entropy_cutoff,
                    white_cutoff, threshold_quantile, cleanse_strength
                )
            elif step == 'd':
                rectified = disjoint_sets_optimization(
                    rectified, cluster_threshold
                )
        end_time = time()
        elapsed_time = end_time - start_time
        
        print("Image transformed correctly [execution time: {:.3f}s]".format(elapsed_time))
        Image.fromarray(rectified, mode="L").save(new_name)
