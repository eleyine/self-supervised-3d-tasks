from pathlib import Path

from self_supervised_3d_tasks.data.generator_base import DataGeneratorBase
import numpy as np

class Numpy2DLoader(DataGeneratorBase):

    def __init__(self,
                 data_path,
                 file_list,
                 batch_size=32,
                 shuffle=False,
                 pre_proc_func=None,
                 n_classes = 2):
        self.n_classes = n_classes
        self.path_to_data = data_path
        self.label_dir = data_path + "_labels"

        if not Path(self.label_dir).exists():
            self.label_dir = None

        super(Numpy2DLoader, self).__init__(file_list, batch_size, shuffle, pre_proc_func)

    def data_generation(self, list_files_temp):
        data_x = []
        data_y = []

        for file_name in list_files_temp:
            path_to_image = "{}/{}".format(self.path_to_data, file_name)
            # print('path_to_image: {}'.format(path_to_image))

            try:
                if self.label_dir:
                    path_label = Path("{}/{}".format(self.label_dir, file_name))
                    path_label = path_label.with_name(path_label.stem).with_suffix(path_label.suffix)
                    # print('path_label: {}'.format(path_label))
                    mask = np.load(path_label)

                path_to_image = "{}/{}".format(self.path_to_data, file_name)
                img = np.load(path_to_image)

                data_x.append(img)

                if self.label_dir:
                    data_y.append(mask)
                else:
                    data_y.append(0)

            except Exception as e:
                print("Error while loading image {}.".format(path_to_image))
                continue
        try:
            data_x = np.stack(data_x)
        except:
            print(data_x.shape)
            # print(data_x[0,0,0])
        data_y = np.stack(data_y)

        if self.label_dir:
            data_y = np.rint(data_y).astype(np.int)
            data_y = np.eye(self.n_classes)[data_y]
            data_y = np.squeeze(data_y, axis=-2)  # remove second last axis, which is still 1

        # Make sure there are no None values
        assert not np.isnan(data_x).any()
        assert not np.isnan(data_y).any()

        return data_x, data_y