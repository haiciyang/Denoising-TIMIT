class data_build(Dataset):
    def __init__(self, trX, trY, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.trX = trX
        self.trY = trY
        self.transform = transform

    def __len__(self):
        return len(trX)

    def __getitem__(self, idx):
        
        sample = {'trX_sample': trX[idx], 'trY_sample': trY[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample