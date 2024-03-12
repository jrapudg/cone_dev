import h5py
import os
import numpy as np
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")


class CarlaH5Dataset(Dataset):
    def __init__(self, dset_path, k_attr, map_attr, split_name="train", mask=True, road_lanes=False):
        self.mask = mask
        self.road_lanes = road_lanes
        
        self.data_root = dset_path
        self.split_name = split_name
        self.pred_horizon = 12
        self.num_others = 7
        self.map_attr = map_attr
        self.predict_yaw = False

        self.k_attr = k_attr

        dataset = h5py.File(os.path.join(self.data_root, self.split_name + '_dataset.hdf5'), 'r')
        self.dset_len = len(dataset["ego_in"])

    def mirror_scene(self, ego_in, ego_out, scene_ids, road_pts, constr_pts):
        ego_in[:,0] = -ego_in[:,0]
        ego_out[:,0] = -ego_out[:,0]
        road_pts[:,:,0] = -road_pts[:,:,0]
        constr_pts[:,:,0] = -constr_pts[:,:,0]
        scene_ids = np.char.add(scene_ids, b'_m')
        return ego_in, ego_out, scene_ids, road_pts, constr_pts


    def __getitem__(self, idx: int):
        dataset = h5py.File(os.path.join(self.data_root, self.split_name + '_dataset.hdf5'), 'r')
        ego_in = dataset['ego_in'][idx]
        ego_out = dataset['ego_out'][idx]
        road_pts = dataset['road_pts'][idx]
        constr_pts = dataset['constr_pts'][idx]    
        scene_ids = dataset['scene_ids'][idx]

        features_mask = np.ones(ego_in.shape[-1], dtype=bool)  # Start with all True

        if self.map_attr == 3:
            features_mask[[2,3,4,5]] = False  # x, y
        elif self.map_attr == 5: 
            features_mask[[2,5]] = False  # x, y, w, l
        elif self.map_attr == 6:
            features_mask[[5]] = False   # x, y, z, w, l
        elif self.map_attr == 7:
            #features_mask[[5]] = False   # x, y, z, w, l
            pass
        else:
            raise NotImplementedError
        
        is_all_zeros_np = np.all(constr_pts == 0)
        if is_all_zeros_np:
            constr_pts[0, 0, :] = np.array([100, 100, 0, 0, 0, 0, 0, 1])

        ego_in = ego_in[:,features_mask]
        ego_out = ego_out[:,features_mask]
        road_pts = road_pts[:,:,features_mask]
        constr_pts = constr_pts[:,:,features_mask]

        

        if "train" in self.split_name:
            should_we_mirror = np.random.choice([0, 1])
            if should_we_mirror:
                ego_in, ego_out, scene_ids, road_pts, constr_pts = self.mirror_scene(ego_in, ego_out, scene_ids, road_pts, constr_pts)
            
        return ego_in, ego_out, road_pts, constr_pts
    def __len__(self):
        return self.dset_len
