import toml 
import xarray as xr
from typing import Literal

class Config:
    def __init__(self, config_path: str, mode: Literal['train', 'predict']):
        self.config = toml.load(config_path)
        self.mode = mode
        
        self.load_data_config()
        self.load_model_config()
        if self.mode == 'train':
            self.load_training_config()
        elif self.mode == 'predict':
            self.load_prediction_config()
        self.load_sys_mode_config()
        
    def load_data_config(self):
        dataset = self.config['dataset']
        folder = self.config['folder']
        TVARs = dataset['TVARs']
        CVARs = dataset['CVARs']
        mask = dataset['mask']
        
        self.TVARs = {
            t['name']: xr.open_dataset(t['path'])[t['variable']]
            for t in TVARs
        }
        self.CVARs = {
            c['name']: xr.open_dataset(c['path'])[c['variable']]
            for c in CVARs
        }
        self.mask = xr.open_dataset(mask['path'])[mask['variable']].values
        self.total_tasks = mask['total']
        self.model_folder = folder['model']
        self.eval_folder = folder['eval']
        self.pred_folder = folder['pred']
        
    def load_model_config(self):
        model_cfg = self.config['model']
        self.hidden_dim = model_cfg['hidden_dim']
        self.n_layers = model_cfg['n_layers']
        self.batch_size = model_cfg['batch_size']
        self.seq_length = model_cfg['seq_length']
        self.window_size = model_cfg['window_size']
        
    def load_training_config(self):
        training_cfg = self.config['train']
        self.start_date = training_cfg['start_date']
        self.end_date = training_cfg['end_date']
        self.lr = training_cfg['lr']
        self.n_epochs = training_cfg['n_epochs']
        self.patience = training_cfg['patience']
        self.verbose_epoch = training_cfg['verbose_epoch']
        self.train_years = training_cfg['train_years']
        
    def load_prediction_config(self):
        prediction_cfg = self.config['prediction']
        self.start_date = prediction_cfg['start_date']
        self.end_date = prediction_cfg['end_date']
        
    def load_sys_mode_config(self):
        sys_mode_cfg = self.config['sys_mode']
        self.debug = sys_mode_cfg['debug']
        self.cover_exist = sys_mode_cfg['cover_exist']
        if self.mode == 'train':
            self.tasks_per_thread = sys_mode_cfg['train_tasks_per_thread']
        elif self.mode == 'predict':
            self.tasks_per_thread = sys_mode_cfg['predict_tasks_per_thread']

if __name__ == "__main__":
    config = Config("config/E.toml", mode='train')