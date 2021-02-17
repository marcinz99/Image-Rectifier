import sys
import numpy as np
from PIL import Image

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
    median = quick_median(array[x_from:x_to, y_from:y_to])
    scale = 1.1 / median
    
    array[x_from:x_to, y_from:y_to] =\
        np.minimum(255*np.power(scale*array[x_from:x_to, y_from:y_to], 4), 255).astype(np.uint8)

transformers = {
    'median_contrast': median_transform_high_contrast,
    'median_basic':    median_transform_basic,
    'show_boxes':      None
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
    return Image.fromarray(img_array, mode="L")



if __name__ == '__main__':
    # Filename! Not filepath!
    filename = str(sys.argv[1]) if len(sys.argv) > 1 else None
    
    if not filename:
        print("Specify the filename (not filepath!) of the image as the first argument.")
    
    else:
        # === SETUP ===========================================================
        # Feel free to hardcode the desirable options
        transformer = transformers['median_contrast']
        split_factor = 1000
        maxdepth = 150
        inplace = False # If true, enhanced photo overwrites the existing one
        prefix = "_"
        # =====================================================================
        
        rectified = rectifier(filename, transformer, maxdepth, split_factor)
        new_name = filename if inplace else prefix + filename
        
        rectified.save(new_name)
