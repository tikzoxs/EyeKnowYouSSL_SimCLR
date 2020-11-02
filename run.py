from simclr import SimCLR
import yaml
from data_utils.dataset_wrapper import DataSetWrapper


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader) # This is a config file that consist of all the paramters neeed config the data loader.

    dataset = DataSetWrapper(config['batch_size'], **config['dataset']) #This is the dataloader object in pytorch.  Refer to data_aug/dataset_wrapper.py

    simclr = SimCLR(dataset, config) #the model and stuff
    simclr.train()


if __name__ == "__main__":
    main()
