import os
import re
import random
from collections import defaultdict


def main():
    # ============ Settings ============
    # Fixed random seed for reproducible spatial partitioning
    seed = 11

    # Number of spatial locations assigned to each split
    # SPO Illegal Building: 2772 train / 1200 test spatial patches
    # DynamicEarthNet:       627  train / 220  test spatial patches
    num_train_spatial = 2772
    num_test_spatial = 1200
  
    root_dir = os.path.dirname(os.path.abspath("E:\TCDMamba\Dataset\Temporal_Illegal_Building"))
    img_dir = os.path.join(root_dir, 'Temporal_Illegal_Building')
    train_info = open(os.path.join(root_dir, 'train_list.txt'), 'w')
    test_info = open(os.path.join(root_dir, 'test_list.txt'), 'w')

# ============ Load and filter TIF files ============
    # Sort the file list first so that the input order is deterministic
    img_list = sorted(os.listdir(img_dir))
    valid_list = []
    for it in img_list:
        if os.path.splitext(it)[1].lower() == '.tif':
            valid_list.append(it)

    # ============ Group patches by spatial-location ID ============
    # File name format: spatialID-temporalID.tif, e.g. 1-3.tif
    # Store (int(tid), filename) so that patches can be sorted numerically
    # by temporal ID. This avoids the lexicographic order "1, 10, 2, ...".
    spatial_dict = defaultdict(list)
    pattern = re.compile(r'^(\d+)-(\d+)\.tif$')
    for it in valid_list:
        m = pattern.match(it)
        if m is None:
            raise ValueError('Bad filename: %s' % it)
        sid = m.group(1)
        tid = int(m.group(2))
        spatial_dict[sid].append((tid, it))

    # Sort each spatial group by temporal ID in numerical order
    for sid in spatial_dict:
        spatial_dict[sid].sort(key=lambda x: x[0])
        spatial_dict[sid] = [x[1] for x in spatial_dict[sid]]

    spatial_ids = sorted(spatial_dict.keys(), key=lambda x: int(x))
    num_spatial = len(spatial_ids)
    print('num_spatial: %d, num_train_spatial: %d, num_test_spatial: %d.'
          % (num_spatial, num_train_spatial, num_test_spatial))

    # ============ Shuffle spatial IDs with a fixed seed ============
    # The partition is performed at the spatial-location level rather than
    # at the patch level, which prevents spatial leakage across splits.
    random.seed(seed)
    random.shuffle(spatial_ids)

    train_ids = spatial_ids[:num_train_spatial]
    test_ids = spatial_ids[num_train_spatial:num_train_spatial + num_test_spatial]

    # ============ Assign all temporal patches of a location to one split ============
    # Because the split is defined by spatial IDs, all temporal patches at the
    # same spatial location go to the same split, and the training and test
    # patches are guaranteed to be spatially disjoint.
    # Patches within each spatial location are already sorted by temporal ID.
    train_list = []
    test_list = []
    for sid in train_ids:
        train_list.extend(spatial_dict[sid])
    for sid in test_ids:
        test_list.extend(spatial_dict[sid])

    print('num_pics: %d, num_train: %d, num_test: %d.'
          % (len(valid_list), len(train_list), len(test_list)))

    # ============ Write split list files ============
    for it in train_list:
        train_info.write(it + '\n')
    for it in test_list:
        test_info.write(it + '\n')

    train_info.close()
    test_info.close()


if __name__ == '__main__':
    main()