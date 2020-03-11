'''
Author: G. L. Roberts
Date: May 2019

About: EDA of data

TODO: Build a dataframe/text document to log all previous scores
'''

class Explore_Data(object):
    def __init__(self, print_stats=False, show_ims=False):
        if print_stats:
            self.count_test_train()
            self.unique_im_sizes()
        if show_ims:
            self.show_class_examples()

    def count_test_train(self):
        train_lst = os.listdir(os.path.join(DATA_DIR, 'train'))
        print(f"No train classes: {len(train_lst)}")
        tot_train = 0
        for ii, fold in enumerate(train_lst):
            class_lst = os.listdir(os.path.join(DATA_DIR, 'train', fold))
            print(f"No. examples in fold {ii}: {len(class_lst)}")
            tot_train += len(class_lst)
        print(f"No train images: {tot_train}")

        test_lst = os.listdir(os.path.join(DATA_DIR, 'test'))
        print(f"No test images: {len(test_lst)}")

    def unique_im_sizes(self):
        train_dir = os.path.join(DATA_DIR, 'train')
        train_folds = os.listdir(train_dir)
        sizes = set()
        for fold in train_folds:
            class_dir = os.path.join(train_dir, fold)
            im_paths = os.listdir(class_dir)
            for im_path in im_paths:
                im = Image.open(os.path.join(class_dir, im_path))
                width, height = im.size
                channels = len(im.mode)
                sizes.add((width, height, channels))
        print(f"Unique image sizes: {sizes}")

    def show_class_examples(self):
        train_dir = os.path.join(DATA_DIR, 'train')
        train_folds = os.listdir(train_dir)
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        for ax, train_set in zip(axes.reshape(-1), train_folds):
            im_paths = os.listdir(os.path.join(train_dir, train_set))
            im_path = random.choice(im_paths)
            img = mpimg.imread(os.path.join(train_dir, train_set, im_path))
            ax.set_title(f"Train set: {train_set}")
            ax.imshow(img)
        plt.tight_layout()
        plt.show()
