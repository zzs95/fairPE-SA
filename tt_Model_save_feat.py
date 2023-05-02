from torchtuples.base import *
from file_and_folder_operations import maybe_mkdir_p

class ttModel(Model):
    
    def init_save_feat(self, exp_dir=''):
        self.exp_dir = exp_dir
        self.save_feat_dir = os.path.join(self.exp_dir, 'saved_feats_surv')
        maybe_mkdir_p(self.save_feat_dir)
        
    def save_feat(self, feat, save_feat_name):
        if isinstance(feat, torch.Tensor):
            feat_np = feat.detach().cpu().numpy()
        else:
            feat_np = feat
        np.save(os.path.join(self.save_feat_dir, save_feat_name+'.npy'), feat_np)
        return
    
    def get_feat(self):
        features_in_hook = []
        features_out_hook = []

        def hook(module, fea_in, fea_out):
            features_in_hook.append(fea_in)
            features_out_hook.append(fea_out)
            return None
        # feature input to fc
        layer_name = [name for (name, module) in self.net.named_modules()][-2] 
        for (name, module) in self.net.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook=hook)
        return features_in_hook
    
    def predict(
        self,
        input,
        batch_size=8224,
        numpy=None,
        eval_=True,
        grads=False,
        to_cpu=False,
        num_workers=0,
        is_dataloader=None,
        func=None,
        save_feat_name=None,
        **kwargs,
    ):
        feat_before_fc = self.get_feat()
        if not hasattr(self.net, "predict"):
            clas_preds = self.predict_net(
                input,
                batch_size,
                numpy,
                eval_,
                grads,
                to_cpu,
                num_workers,
                is_dataloader,
                func,
                **kwargs,
            )
        else:
            pred_func = wrapfunc(func, self.net.predict)
            clas_preds = self._predict_func(
                pred_func,
                input,
                batch_size,
                numpy,
                eval_,
                grads,
                to_cpu,
                num_workers,
                is_dataloader,
                **kwargs,
            )
        
        feat = feat_before_fc[0][0]
        self.save_feat(feat, save_feat_name)
            
        return array_or_tensor(clas_preds, numpy, input)