
def get_list(opt, split, path, subsets, category_dict, data_percentage, max_imgs):
    data_list = []
    for subset in subsets:
        for cat in category_dict[subset]:
            list_fname = f"{path}/{subset}/lists/{cat}_{split}.list"
            if not os.path.exists(list_fname):
                continue
            lines = open(list_fname).read().splitlines()
            lines = lines[:round(data_percentage*len(lines))]
            for i, img_fname in enumerate(lines):
                if i >= max_imgs: break
                name = '.'.join(img_fname.split('.')[:-1])
                object_name = name.split('_')[-2]
                sample_id = name.split('_')[-1]
                data_list.append((subset, cat, object_name, sample_id))
    return data_list

# def get_list(self, opt, split):
#     data_list = []
#     for subset in self.subsets:
#         for cat in self.category_dict[subset]:
#             list_fname = f"{self.path}/{subset}/lists/{cat}_{split}.list"
#             if not os.path.exists(list_fname):
#                 continue
#             lines = open(list_fname).read().splitlines()
#             lines = lines[:round(self.data_percentage*len(lines))]
#             for i, img_fname in enumerate(lines):
#                 if i >= self.max_imgs: break
#                 name = '.'.join(img_fname.split('.')[:-1])
#                 object_name = name.split('_')[-2]
#                 sample_id = name.split('_')[-1]
#                 data_list.append((subset, cat, object_name, sample_id))
#     return data_list


def get_camera(subset, category, object_name, sample_id):
    fname = f"{category}/{category}_{object_name}_{sample_id}"
    intr_p = f"{path}/{subset}/camera_data/intr/{fname}.npy"
    extr_p = f"{path}/{subset}/camera_data/extr/{fname}.npy"
    Rt = np.load(extr_p)
    K = torch.from_numpy(np.load(intr_p))
    return K, Rt

def get_gt_sdf(subset, category, object_name):
    fname = f"{category}/{category}_{object_name}"
    gt_fname = f"{path}/{subset}/gt_sdf/{fname}.npy"
    gt_dict = np.load(gt_fname, allow_pickle=True).item()
    gt_sample_points = torch.from_numpy(gt_dict['sample_pt']).float()
    gt_sample_sdf = torch.from_numpy(gt_dict['sample_sdf']).float() - 0.003
    return gt_sample_points, gt_sample_sdf

if __name__ == "__main__":

    path = '/home/zubairirshad/Downloads/train_data'
    category_dict = {}
    category_list = []

    max_imgs = 10
    data_percentage = 1
    
    subsets = ['objaverse_LVIS_tiny']
    for subset in subsets:
        subset_path = "{}/{}".format(path, subset)
        categories = [name[:-11] for name in os.listdir("{}/lists".format(subset_path)) if name.endswith("_train.list")]
        category_dict[subset] = categories
        category_list += [cat for cat in categories]

    lists_all = get_list(opt, 'train', subsets, category_dict, data_percentage, max_imgs)

    categories = [name[:-11] for name in os.listdir("{}/lists".format(subset_path)) if name.endswith("_train.list")]

    subset, category, object_name, sample_id = list[idx]