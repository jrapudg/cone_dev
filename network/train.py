import os
import numpy as np
import random

import torch
import torch.distributions as D
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from utils.metric_helpers import min_xde_K
from utils.train_helpers import nll_loss_multimodes

from utils.dataset import CarlaH5Dataset
from models.cone_model import ConstructEgo
#from models.static_former import ConstructEgo

class Trainer:
    def __init__(self, args, results_dirname):
        self.args = args
        self.results_dirname = results_dirname
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available() and not self.args.disable_cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.args.seed)
        else:
            self.device = torch.device("cpu")

        self.initialize_dataloaders()
        self.initialize_model()
        self.optimiser = optim.Adam(self.autobot_model.parameters(), lr=self.args.learning_rate,
                                    eps=self.args.adam_epsilon)
        self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=args.learning_rate_sched, gamma=0.5,
                                               verbose=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.results_dirname, "tb_files"))
        self.smallest_minade_k = 5.0  # for computing best models
        self.smallest_minfde_k = 5.0  # for computing best models

    def initialize_dataloaders(self):
        if "carla" in self.args.dataset:
            train_dset = CarlaH5Dataset(dset_path=self.args.dataset_path, k_attr=self.args.k_attr,
                                        map_attr=self.args.map_attr, split_name="train")
            val_dset = CarlaH5Dataset(dset_path=self.args.dataset_path, k_attr=self.args.k_attr,
                                        map_attr=self.args.map_attr, split_name="val")
        else:
            raise NotImplementedError
        
        self.num_other_agents = train_dset.num_others
        self.pred_horizon = train_dset.pred_horizon
        self.k_attr = train_dset.k_attr
        self.map_attr = train_dset.map_attr

        self.train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=self.args.batch_size, shuffle=True, num_workers=12, drop_last=False, pin_memory=False
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dset, batch_size=self.args.batch_size, shuffle=True, num_workers=12, drop_last=False, pin_memory=False
        )

        print("Train dataset loaded with length", len(train_dset))
        print("Val dataset loaded with length", len(val_dset))

    def initialize_model(self):
        if "Ego" in self.args.model_type:
            self.autobot_model = ConstructEgo(k_attr=self.k_attr,
                                              d_k=self.args.hidden_size,
                                              _M=self.num_other_agents,
                                              c=self.args.num_modes,
                                              T=self.pred_horizon,
                                              L_enc=self.args.num_encoder_layers,
                                              dropout=self.args.dropout,
                                              num_heads=self.args.tx_num_heads,
                                              L_dec=self.args.num_decoder_layers,
                                              tx_hidden_size=self.args.tx_hidden_size,
                                              map_attr=self.map_attr,
                                              construction_attention=self.args.construction_attention, 
                                              sequence_attention=self.args.sequence_attention,
                                              baseline =self.args.baseline).to(self.device)
        else:
            raise NotImplementedError
    
    def _data_to_device(self, data):
        if "Ego" in self.args.model_type:
            ego_in, ego_out, roads, constr = data
            #roads_t, con_t = parse_roads(roads, constr)
            ego_in = ego_in.float().to(self.device)
            ego_out = ego_out.float().to(self.device)
            roads = roads.float().to(self.device)
            constr = constr.float().to(self.device)
            return ego_in, ego_out, roads, constr

    def _compute_ego_errors(self, ego_preds, ego_gt):
        ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
        ade_losses = torch.mean(torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1), dim=1).transpose(0, 1).cpu().numpy()
        fde_losses = torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1).transpose(0, 1).cpu().numpy()
        return ade_losses, fde_losses

    def constructego_train(self):
        steps = 0
        for epoch in range(0, self.args.num_epochs):
            print("Epoch:", epoch)
            epoch_ade_losses = []
            epoch_fde_losses = []
            epoch_mode_probs = []
            for i, data in enumerate(self.train_loader):
                ego_in, ego_out, roads_t, construction_t = self._data_to_device(data)
                pred_obs, mode_probs = self.autobot_model(ego_in, roads_t, construction_t)

                nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, ego_out[:, :, :2], mode_probs,
                                                                                   entropy_weight=self.args.entropy_weight,
                                                                                   kl_weight=self.args.kl_weight,
                                                                                   use_FDEADE_aux_loss=self.args.use_FDEADE_aux_loss)

                self.optimiser.zero_grad()
                (nll_loss + adefde_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.autobot_model.parameters(), self.args.grad_clip_norm)
                self.optimiser.step()

                self.writer.add_scalar("Loss/nll", nll_loss.item(), steps)
                self.writer.add_scalar("Loss/adefde", adefde_loss.item(), steps)
                self.writer.add_scalar("Loss/kl", kl_loss.item(), steps)

                with torch.no_grad():
                    ade_losses, fde_losses = self._compute_ego_errors(pred_obs, ego_out)
                    epoch_ade_losses.append(ade_losses)
                    epoch_fde_losses.append(fde_losses)
                    epoch_mode_probs.append(mode_probs.detach().cpu().numpy())

                if i % 10 == 0:
                    print(i, "/", len(self.train_loader.dataset)//self.args.batch_size,
                          "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                          "Prior Entropy", round(torch.mean(D.Categorical(mode_probs).entropy()).item(), 2),
                          "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2))

                steps += 1

            ade_losses = np.concatenate(epoch_ade_losses)
            fde_losses = np.concatenate(epoch_fde_losses)
            mode_probs = np.concatenate(epoch_mode_probs)

            train_minade_c = min_xde_K(ade_losses, mode_probs, K=self.args.num_modes)
            train_minade_10 = min_xde_K(ade_losses, mode_probs, K=min(self.args.num_modes, 10))
            train_minade_5 = min_xde_K(ade_losses, mode_probs, K=min(self.args.num_modes, 5))
            train_minade_1 = min_xde_K(ade_losses, mode_probs, K=1)
            train_minfde_c = min_xde_K(fde_losses, mode_probs, K=min(self.args.num_modes, 10))
            train_minfde_1 = min_xde_K(fde_losses, mode_probs, K=1)
            print("Train minADE c:", train_minade_c[0], "Train minADE 1:", train_minade_1[0], "Train minFDE c:", train_minfde_c[0])

            # Log train metrics
            self.writer.add_scalar("metrics/Train minADE_{}".format(self.args.num_modes), train_minade_c[0], epoch)
            self.writer.add_scalar("metrics/Train minADE_{}".format(10), train_minade_10[0], epoch)
            self.writer.add_scalar("metrics/Train minADE_{}".format(5), train_minade_5[0], epoch)
            self.writer.add_scalar("metrics/Train minADE_{}".format(1), train_minade_1[0], epoch)
            self.writer.add_scalar("metrics/Train minFDE_{}".format(self.args.num_modes), train_minfde_c[0], epoch)
            self.writer.add_scalar("metrics/Train minFDE_{}".format(1), train_minfde_1[0], epoch)

            # update learning rate
            self.optimiser_scheduler.step()

            self.constructego_evaluate(epoch)
            self.save_model(epoch)
            print("Best minADE c", self.smallest_minade_k, "Best minFDE c", self.smallest_minfde_k)

    def constructego_evaluate(self, epoch):
        self.autobot_model.eval()
        with torch.no_grad():
            val_ade_losses = []
            val_fde_losses = []
            val_mode_probs = []
            for i, data in enumerate(self.val_loader):
                ego_in, ego_out, roads_t, constructions_t = self._data_to_device(data)

                # encode observations
                pred_obs, mode_probs = self.autobot_model(ego_in, roads_t, constructions_t)

                ade_losses, fde_losses = self._compute_ego_errors(pred_obs, ego_out)
                val_ade_losses.append(ade_losses)
                val_fde_losses.append(fde_losses)
                val_mode_probs.append(mode_probs.detach().cpu().numpy())

            val_ade_losses = np.concatenate(val_ade_losses)
            val_fde_losses = np.concatenate(val_fde_losses)
            val_mode_probs = np.concatenate(val_mode_probs)
            val_minade_c = min_xde_K(val_ade_losses, val_mode_probs, K=self.args.num_modes)
            val_minade_10 = min_xde_K(val_ade_losses, val_mode_probs, K=min(self.args.num_modes, 10))
            val_minade_5 = min_xde_K(val_ade_losses, val_mode_probs, K=5)
            val_minade_1 = min_xde_K(val_ade_losses, val_mode_probs, K=1)
            val_minfde_c = min_xde_K(val_fde_losses, val_mode_probs, K=self.args.num_modes)
            val_minfde_1 = min_xde_K(val_fde_losses, val_mode_probs, K=1)

            # Log val metrics
            self.writer.add_scalar("metrics/Val minADE_{}".format(self.args.num_modes), val_minade_c[0], epoch)
            self.writer.add_scalar("metrics/Val minADE_{}".format(10), val_minade_10[0], epoch)
            self.writer.add_scalar("metrics/Val minADE_{}".format(5), val_minade_5[0], epoch)
            self.writer.add_scalar("metrics/Val minADE_{}".format(1), val_minade_1[0], epoch)
            self.writer.add_scalar("metrics/Val minFDE_{}".format(self.args.num_modes), val_minfde_c[0], epoch)
            self.writer.add_scalar("metrics/Val minFDE_{}".format(1), val_minfde_1[0], epoch)

            print("minADE c:", val_minade_c[0], "minADE_10", val_minade_10[0], "minADE_5", val_minade_5[0],
                  "minFDE c:", val_minfde_c[0], "minFDE_1:", val_minfde_1[0])
            self.autobot_model.train()
            self.save_model(minade_k=val_minade_c[0], minfde_k=val_minfde_c[0])

    def save_model(self, epoch=None, minade_k=None, minfde_k=None):
        if epoch is None:
            if minade_k < self.smallest_minade_k:
                self.smallest_minade_k = minade_k
                torch.save(
                    {
                        "AutoBot": self.autobot_model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "best_models_ade.pth"),
                )

            if minfde_k < self.smallest_minfde_k:
                self.smallest_minfde_k = minfde_k
                torch.save(
                    {
                        "AutoBot": self.autobot_model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "best_models_fde.pth"),
                )

        else:
            if epoch % 10 == 0 and epoch > 0:
                torch.save(
                    {
                        "AutoBot": self.autobot_model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "models_%d.pth" % epoch),
                )

    def train(self):
        if "Ego" in self.args.model_type:
            self.constructego_train()
        else:
            raise NotImplementedError

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

if __name__ == "__main__":
    args = Args(exp_id="inv_prev",
                disable_cuda=False, 
                seed=1, 
                learning_rate=0.00075,
                adam_epsilon=1e-4,
                learning_rate_sched=[10, 20, 30, 40, 50],
                dataset="carla",
                save_dir="/home/juan/Documents/simulators/cone_sim/",
                dataset_path="/home/juan/Documents/simulators/cone_sim/dataset_bp/",
                model_type="Cone-Ego",
                batch_size=64,
                hidden_size=128,
                num_modes=10,
                num_encoder_layers=2,
                dropout=0.1,
                tx_num_heads=16,
                num_decoder_layers=2,
                tx_hidden_size=384,
                num_epochs=150,
                entropy_weight=40.0,
                kl_weight=20.0,
                use_FDEADE_aux_loss=True,
                grad_clip_norm = 5.0,
                map_attr = 7, #7
                k_attr = 6, #6
                construction_attention=False, 
                sequence_attention= False,
                baseline = True
                )

    model_configname = ""
    model_configname += "Cone_ego"
    model_configname += "_C"+str(args.num_modes) + "_H"+str(args.hidden_size) + "_E"+str(args.num_encoder_layers)
    model_configname += "_D"+str(args.num_decoder_layers) + "_TXH"+str(args.tx_hidden_size) + "_NH"+str(args.tx_num_heads)
    model_configname += "_EW"+str(int(args.entropy_weight)) + "_KLW"+str(int(args.kl_weight))
    model_configname += "_NormLoss" if args.use_FDEADE_aux_loss else ""
    model_configname += "_Baseline" if args.baseline else ""
    model_configname += "_Cone" if (not args.baseline and args.construction_attention) else ""
    model_configname += "_kattr"+str(args.k_attr)
    model_configname += "_mattr"+str(args.map_attr)


    if args.exp_id is not None:
        model_configname += ("_" + args.exp_id)
    model_configname += "_s"+str(args.seed)

    results_dirname = os.path.join(args.save_dir, "results", args.dataset, model_configname)

    trainer = Trainer(args, results_dirname)
    trainer.train()
