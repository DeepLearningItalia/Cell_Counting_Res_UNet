import pickle


def write_config(cfg: dict, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(cfg, f)
        
def load_config(load_path: str):
    with open(load_path, "rb") as f:
        return pickle.load(f)
