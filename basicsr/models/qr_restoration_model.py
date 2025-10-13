import torch
from torch import nn
from models.image_restoration_model import ImageRestorationModel
from losses import SobelEdgeLoss
from losses import BinaryLoss
from archs import define_network
from utils import get_root_logger

#from basicsr.models.image_restoration_model import ImageRestorationModel
#from basicsr.losses import SobelEdgeLoss
#from basicsr.losses import BinaryLoss


class QRRestorationModel(ImageRestorationModel):
    """
    Model for EGRestormer with QR-specific losses.
    """
    def __init__(self, opt):
        super().__init__(opt)
       
        self.sobel_loss = SobelEdgeLoss().to(self.device)
        self.bce_loss = BinaryLoss().to(self.device)
        #self.sobel_weight = opt['train'].get('sobel_weight', 0.1)
        
        # �������ļ���ȡ���Ȳ���
        train_opt = opt['train']
        self.sw_start = train_opt.get('sobel_weight_start', 0.0)
        self.sw_end = train_opt.get('sobel_weight_end', 0.5)
        self.sw_start_iter = train_opt.get('sobel_weight_start_iter', 10000) # ���磬�ӵ�10000�ε�����ʼ����
        self.sw_end_iter = train_opt.get('sobel_weight_end_iter', 100000) # ���磬�ڵ�100000�ε����ﵽ���ֵ

    
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        '''
        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')
        '''

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        
    def get_current_sobel_weight(self, current_iter):
        """linearly increasing sobel_weight"""
        if current_iter < self.sw_start_iter:
            return self.sw_start
        if current_iter > self.sw_end_iter:
            return self.sw_end

        # ���Բ�ֵ
        progress = (current_iter - self.sw_start_iter) / (self.sw_end_iter - self.sw_start_iter)
        return self.sw_start + progress * (self.sw_end - self.sw_start)

    


        
        
        