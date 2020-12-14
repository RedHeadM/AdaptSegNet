import torch
import numpy as np
import scipy.misc as m
from torch.utils import data
from multiview.video.datasets import ViewPairDataset

def get_voc_palette(num_classes):
    n = num_classes
    palette = [0]*(n*3)
    for j in range(0,n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while (lab > 0):
                    palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return palette

class MulitviewSegLoader(data.Dataset):

    n_classes = 12
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(n_classes), colors))

    mean_rgb = {"stillseg": [115.75946042, 116.16258986, 115.06630661],}

    def __init__(
        self,
        root,
        number_views,
        view_idx,
        num_classes,
        load_seg_mask = True,
        is_transform=True,
        img_size=(300, 300),
        img_norm=True,
        augmentations=None,
        version="stillseg",
        return_id=False,
        img_mean = np.array([115.75946042, 116.16258986, 115.06630661],)
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.palette = get_voc_palette(num_classes)
        self.root = root
        self.is_transform = is_transform
        self.view_idx = view_idx
        self.number_views = number_views
        self.img_size= img_size
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.load_seg_mask= load_seg_mask
        self.mean = img_mean
        print('view_idx: {}'.format(view_idx))
        print('self.img_size: {}'.format(self.img_size))

        if not isinstance(view_idx,int):
            raise ValueError('view_idx: {}'.format(view_idx))
        self.view_key_img = "frames views " + str(self.view_idx)
        self.view_key_seg = "seg "+str(self.view_idx)
        # assert isinstance(view_idx, int) and isinstance(number_views, int) self.img_size = (
            # img_size if isinstance(img_size, tuple) else (img_size, img_size)
        # )
        self.void_classes = []
        # obi robot backgour and 10 obj
        self.valid_classes = list(range(self.n_classes))
        self.class_names = [
            "background",
            "robo",
            "person",
            "rack",
            "pallet",
            "floor_line",
            "gitterbox",
            "vehicle",
	    "trolley"
	    "dk"
	    "ddlskdj"
	    "klsjadf"
        ]

        # self.ignore_index = 250
        self.ignore_index = 255
        self.split ='view idx {} with segmenation {}'.format(view_idx,load_seg_mask)
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))
        self._set_files()
        print("Found %d %s images" % (len(self), self.split))

        self.return_id = return_id

    def _set_files(self):
        def data_len_filter(comm_name,frame_len_paris):
            if len(frame_len_paris)<2:
                return frame_len_paris[0]>10
            return min(*frame_len_paris)>10
        self.mvbdata = ViewPairDataset(self.root.strip(),
                                        segmentation= True,
                                        # segmentation= self.load_seg_mask, TODO
                                        transform_frames= None,
                                        number_views=self.number_views,
                                        filter_func=data_len_filter)

    def __len__(self):
        """__len__"""
        return len(self.mvbdata)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """

        s = self.mvbdata[index]
        lbl = s[self.view_key_seg]
        img = s[self.view_key_img]

        img = np.asarray(img, dtype=np.float32)
        lbl = np.asarray(lbl, dtype=np.int32)

        # img = np.array(img, dtype=np.uint8)
        # lbl = np.array(lbl, dtype=np.uint8)
        # lbl = self.encode_segmap(lbl)

        # if self.augmentations is not None:
            # img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        img_name = s['common name']

        return img, lbl, img.shape, img_name

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
            # print('nornm img: {}'.format(np.max(img)))
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        # if not np.all(classes == np.unique(lbl)):
            # print("WARN: resizing labels yielded fewer classes")

        # if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            # print("after det", classes, np.unique(lbl))
            # raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


