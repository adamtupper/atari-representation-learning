import torch
from torch import nn
from .utils import EarlyStopping, appendabledict, \
    calculate_multiclass_accuracy, calculate_multiclass_f1_score, \
    calculate_mse, calculate_mape, \
    append_suffix, compute_dict_average

from copy import deepcopy
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .categorization import summary_key_dict


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_outputs=255):
        super().__init__()
        self.model = nn.Linear(in_features=input_dim, out_features=num_outputs)

    def forward(self, feature_vectors):
        return self.model(feature_vectors)


class FullySupervisedLinearProbe(nn.Module):
    def __init__(self, encoder, num_outputs=255):
        super().__init__()
        self.encoder = deepcopy(encoder)
        self.probe = LinearProbe(input_dim=self.encoder.hidden_size,
                                 num_outputs=num_outputs)

    def forward(self, x):
        feature_vec = self.encoder(x)
        return self.probe(feature_vec)


class NonLinearProbe(nn.Module):
    """
    Simple non-linear model with one hidden layer the same size as the input
    layer.
    """
    def __init__(self, input_dim, num_outputs=255):
        super().__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=input_dim),
            nn.ReLU(),
            nn.Linear(in_features=input_dim, out_features=num_outputs),
        )

    def forward(self, feature_vectors):
        return self.model(feature_vectors)


class ProbeTrainer:
    def __init__(self,
                 encoder=None,
                 method_name="my_method",
                 wandb=None,
                 patience=15,
                 num_outputs=256,
                 fully_supervised=False,
                 save_dir=".models",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr=5e-4,
                 epochs=100,
                 batch_size=64,
                 representation_len=256,
                 verbose=True,
                 non_linear=False):

        self.encoder = encoder
        self.wandb = wandb
        self.device = device
        self.fully_supervised = fully_supervised
        self.save_dir = save_dir
        self.num_outputs = num_outputs
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.method = method_name
        self.feature_size = representation_len
        self.loss_fn = nn.CrossEntropyLoss()
        self.verbose = verbose
        self.non_linear = non_linear

        # bad convention, but these get set in "create_probes"
        self.probes = self.early_stoppers = self.optimizers = self.schedulers = None

    def create_probes(self, sample_label):
        if self.fully_supervised:
            assert self.encoder != None, "for fully supervised you must provide an encoder!"
            self.probes = {k: FullySupervisedLinearProbe(encoder=self.encoder,
                                                         num_outputs=self.num_outputs).to(self.device) for k in
                           sample_label.keys()}
        elif self.non_linear:
            self.probes = {k: NonLinearProbe(input_dim=self.feature_size,
                                             num_outputs=self.num_outputs).to(self.device) for k in sample_label.keys()}
        else:
            self.probes = {k: LinearProbe(input_dim=self.feature_size,
                                          num_outputs=self.num_outputs).to(self.device) for k in sample_label.keys()}

        self.early_stoppers = {
            k: EarlyStopping(patience=self.patience, verbose=self.verbose, name=k + "_probe", save_dir=self.save_dir)
            for k in sample_label.keys()}

        self.optimizers = {k: torch.optim.Adam(list(self.probes[k].parameters()),
                                               eps=1e-5, lr=self.lr) for k in sample_label.keys()}
        self.schedulers = {
            k: torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizers[k], patience=5, factor=0.2, verbose=self.verbose,
                                                          mode='max', min_lr=1e-5) for k in sample_label.keys()}

    def generate_batch(self, episodes, episode_labels):
        total_steps = sum([len(e) for e in episodes])
        assert total_steps > self.batch_size
        if self.verbose:
            print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)

        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            episode_labels_batch = [episode_labels[x] for x in indices]
            xs, labels = [], appendabledict()
            for ep_ind, episode in enumerate(episodes_batch):
                # Get one sample from this episode
                t = np.random.randint(len(episode))
                xs.append(episode[t])
                labels.append_update(episode_labels_batch[ep_ind][t])
            yield torch.stack(xs).float().to(self.device) / 255., labels

    def probe(self, batch, k):
        probe = self.probes[k]
        probe.to(self.device)
        if self.fully_supervised:
            # if method is supervised batch is a batch of frames and probe is a full encoder + linear or nonlinear probe
            preds = probe(batch)

        elif not self.encoder:
            # if encoder is None then inputs are vectors
            f = batch.detach()
            assert len(f.squeeze().shape) == 2, "if input is not a batch of vectors you must specify an encoder!"
            preds = probe(f)

        else:
            with torch.no_grad():
                self.encoder.to(self.device)
                f = self.encoder(batch).detach()
            preds = probe(f)
        return preds

    def log_results(self, epoch_idx, *dictionaries):
        if self.verbose:
            print("Epoch: {}".format(epoch_idx))
            for dictionary in dictionaries:
                for k, v in dictionary.items():
                    print("\t {}: {:8.4f}".format(k, v))
                print("\t --")

    # Very little changes between these functions for classification and regression.
    # Would it be better to generalise them and make them ProbeTrainer methods?
    def do_one_epoch(self, episodes, label_dicts):
        raise NotImplementedError("Use either a classification or regression probe trainer.")

    def do_test_epoch(self, episodes, label_dicts):
        raise NotImplementedError("Use either a classification or regression probe trainer.")

    def train(self, tr_eps, val_eps, tr_labels, val_labels):
        raise NotImplementedError("Use either a classification or regression probe trainer.")

    def evaluate(self, val_episodes, val_label_dicts, epoch=None):
        raise NotImplementedError("Use either a classification or regression probe trainer.")

    def test(self, test_episodes, test_label_dicts, epoch=None):
        raise NotImplementedError("Use either a classification or regression probe trainer.")


class ClassificationProbeTrainer(ProbeTrainer):
    def __init__(self,
                 encoder=None,
                 method_name="my_method",
                 wandb=None,
                 patience=15,
                 num_classes=256,
                 fully_supervised=False,
                 save_dir=".models",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr=5e-4,
                 epochs=100,
                 batch_size=64,
                 representation_len=256,
                 verbose=True,
                 non_linear=False):

        super().__init__(encoder, method_name, wandb, patience, num_classes,
                         fully_supervised, save_dir, device, lr, epochs,
                         batch_size, representation_len, verbose, non_linear)
        self.loss_fn = nn.CrossEntropyLoss()

        # bad convention, but these get set in "create_probes"
        self.probes = self.early_stoppers = self.optimizers = self.schedulers = None

    def do_one_epoch(self, episodes, label_dicts):
        sample_label = label_dicts[0][0]
        epoch_loss, accuracy = {k + "_loss": [] for k in sample_label.keys() if
                                not self.early_stoppers[k].early_stop}, \
                               {k + "_acc": [] for k in sample_label.keys() if
                                not self.early_stoppers[k].early_stop}

        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                if self.early_stoppers[k].early_stop:
                    continue
                optim = self.optimizers[k]
                optim.zero_grad()

                label = torch.tensor(label).long().to(self.device)
                preds = self.probe(x, k)

                loss = self.loss_fn(preds, label)

                epoch_loss[k + "_loss"].append(loss.detach().item())
                preds = preds.cpu().detach().numpy()
                preds = np.argmax(preds, axis=1)
                label = label.cpu().detach().numpy()
                accuracy[k + "_acc"].append(calculate_multiclass_accuracy(preds,
                                                                          label))
                if self.probes[k].training:
                    loss.backward()
                    optim.step()

        epoch_loss = {k: np.mean(loss) for k, loss in epoch_loss.items()}
        accuracy = {k: np.mean(acc) for k, acc in accuracy.items()}

        return epoch_loss, accuracy

    def do_test_epoch(self, episodes, label_dicts):
        sample_label = label_dicts[0][0]
        accuracy_dict, f1_score_dict = {}, {}
        pred_dict, all_label_dict = {k: [] for k in sample_label.keys()}, \
                                    {k: [] for k in sample_label.keys()}

        # collect all predictions first
        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                label = torch.tensor(label).long().cpu()
                all_label_dict[k].append(label)
                preds = self.probe(x, k).detach().cpu()
                pred_dict[k].append(preds)

        for k in all_label_dict.keys():
            preds, labels = torch.cat(pred_dict[k]).cpu().detach().numpy(),\
                            torch.cat(all_label_dict[k]).cpu().detach().numpy()

            preds = np.argmax(preds, axis=1)
            accuracy = calculate_multiclass_accuracy(preds, labels)
            f1score = calculate_multiclass_f1_score(preds, labels)
            accuracy_dict[k] = accuracy
            f1_score_dict[k] = f1score

        return accuracy_dict, f1_score_dict

    def train(self, tr_eps, val_eps, tr_labels, val_labels):
        # if not self.encoder:
        #     assert len(tr_eps[0][0].squeeze().shape) == 2, "if input is a batch of vectors you must specify an encoder!"
        sample_label = tr_labels[0][0]
        self.create_probes(sample_label)
        e = 0
        all_probes_stopped = np.all([early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        while (not all_probes_stopped) and e < self.epochs:
            epoch_loss, accuracy = self.do_one_epoch(tr_eps, tr_labels)
            self.log_results(e, epoch_loss, accuracy)

            val_loss, val_accuracy = self.evaluate(val_eps, val_labels, epoch=e)
            # update all early stoppers
            for k in sample_label.keys():
                if not self.early_stoppers[k].early_stop:
                    self.early_stoppers[k](val_accuracy["val_" + k + "_acc"], self.probes[k])

            for k, scheduler in self.schedulers.items():
                if not self.early_stoppers[k].early_stop:
                    scheduler.step(val_accuracy['val_' + k + '_acc'])
            e += 1
            all_probes_stopped = np.all([early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        if self.verbose:
            print("All probes early stopped!")

    def evaluate(self, val_episodes, val_label_dicts, epoch=None):
        for k, probe in self.probes.items():
            probe.eval()
        epoch_loss, accuracy = self.do_one_epoch(val_episodes, val_label_dicts)
        epoch_loss = {"val_" + k: v for k, v in epoch_loss.items()}
        accuracy = {"val_" + k: v for k, v in accuracy.items()}
        self.log_results(epoch, epoch_loss, accuracy)
        for k, probe in self.probes.items():
            probe.train()
        return epoch_loss, accuracy

    def test(self, test_episodes, test_label_dicts, epoch=None):
        for k in self.early_stoppers.keys():
            self.early_stoppers[k].early_stop = False
        for k, probe in self.probes.items():
            probe.eval()
        acc_dict, f1_dict = self.do_test_epoch(test_episodes, test_label_dicts)

        acc_dict, f1_dict = postprocess_raw_metrics(acc_dict, f1_dict)

        if self.verbose:
            print("""In our paper, we report F1 scores and accuracies averaged across each category. 
                That is, we take a mean across all state variables in a category to get the average score for that category.
                Then we average all the category averages to get the final score that we report per game for each method. 
                These scores are called \'across_categories_avg_acc\' and \'across_categories_avg_f1\' respectively
                We do this to prevent categories with large number of state variables dominating the mean F1 score.
                """)
        self.log_results("Test", acc_dict, f1_dict)
        return acc_dict, f1_dict


class RegressionProbeTrainer(ProbeTrainer):
    def __init__(self,
                 encoder=None,
                 method_name="my_method",
                 wandb=None,
                 patience=15,
                 fully_supervised=False,
                 save_dir=".models",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr=5e-4,
                 epochs=100,
                 batch_size=64,
                 representation_len=256,
                 verbose=True,
                 non_linear=False):

        super().__init__(encoder, method_name, wandb, patience, 1,
                         fully_supervised, save_dir, device, lr, epochs,
                         batch_size, representation_len, verbose, non_linear)
        self.loss_fn = nn.MSELoss()

        # bad convention, but these get set in "create_probes"
        self.probes = self.early_stoppers = self.optimizers = self.schedulers = None

    def do_one_epoch(self, episodes, label_dicts):
        sample_label = label_dicts[0][0]
        epoch_loss, mse = {k + "_loss": [] for k in sample_label.keys() if not self.early_stoppers[k].early_stop}, \
                          {k + "_mse": [] for k in sample_label.keys() if not self.early_stoppers[k].early_stop}

        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                if self.early_stoppers[k].early_stop:
                    continue
                optim = self.optimizers[k]
                optim.zero_grad()

                label = torch.tensor(label).float().to(self.device)
                preds = self.probe(x, k)

                loss = self.loss_fn(preds, label)

                epoch_loss[k + "_loss"].append(loss.detach().item())
                preds = preds.cpu().detach().numpy()
                # preds = np.argmax(preds, axis=1)
                label = label.cpu().detach().numpy()
                mse[k + "_mse"].append(calculate_mse(preds, label))  # TODO: Does MSE need to be calculated separately to the loss?
                if self.probes[k].training:
                    loss.backward()
                    optim.step()

        epoch_loss = {k: np.mean(loss) for k, loss in epoch_loss.items()}
        mse = {k: np.mean(mse) for k, mse in mse.items()}

        return epoch_loss, mse

    def do_test_epoch(self, episodes, label_dicts):
        # TODO: Decide on performance metrics
        sample_label = label_dicts[0][0]
        mse_dict, mape_dict = {}, {}
        pred_dict, all_label_dict = {k: [] for k in sample_label.keys()}, \
                                    {k: [] for k in sample_label.keys()}

        # collect all predictions first
        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                label = torch.tensor(label).float().cpu()
                all_label_dict[k].append(label)
                preds = self.probe(x, k).detach().cpu()
                pred_dict[k].append(preds)

        for k in all_label_dict.keys():
            preds, labels = torch.cat(pred_dict[k]).cpu().detach().numpy(), \
                            torch.cat(all_label_dict[k]).cpu().detach().numpy()

            # preds = np.argmax(preds, axis=1)
            mse = calculate_mse(preds, labels)
            mape = calculate_mape(preds, labels, offset=1)  # Make offset a configurable parameter
            mse_dict[k] = mse
            mape_dict[k] = mape

        return mse_dict, mape_dict

    def train(self, tr_eps, val_eps, tr_labels, val_labels):
        sample_label = tr_labels[0][0]
        self.create_probes(sample_label)
        e = 0
        all_probes_stopped = np.all([early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        while (not all_probes_stopped) and e < self.epochs:
            epoch_loss, mse = self.do_one_epoch(tr_eps, tr_labels)
            self.log_results(e, epoch_loss, mse)

            val_loss, val_mse = self.evaluate(val_eps, val_labels, epoch=e)
            # update all early stoppers
            for k in sample_label.keys():
                if not self.early_stoppers[k].early_stop:
                    # Use -ve MSE as higher is supposed to be better
                    self.early_stoppers[k](-val_mse["val_" + k + "_mse"], self.probes[k])

            for k, scheduler in self.schedulers.items():
                if not self.early_stoppers[k].early_stop:
                    scheduler.step(val_mse['val_' + k + '_mse'])
            e += 1
            all_probes_stopped = np.all([early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        if self.verbose:
            print("All probes early stopped!")

    def evaluate(self, val_episodes, val_label_dicts, epoch=None):
        for k, probe in self.probes.items():
            probe.eval()
        epoch_loss, mse = self.do_one_epoch(val_episodes, val_label_dicts)
        epoch_loss = {"val_" + k: v for k, v in epoch_loss.items()}
        mse = {"val_" + k: v for k, v in mse.items()}
        self.log_results(epoch, epoch_loss, mse)
        for k, probe in self.probes.items():
            probe.train()
        return epoch_loss, mse

    def test(self, test_episodes, test_label_dicts, epoch=None):
        for k in self.early_stoppers.keys():
            self.early_stoppers[k].early_stop = False
        for k, probe in self.probes.items():
            probe.eval()
        mse_dict, mape_dict = self.do_test_epoch(test_episodes, test_label_dicts)

        mse_dict, mape_dict = postprocess_raw_metrics(mse_dict, mape_dict)

        self.log_results("Test", mse_dict) #, mape_dict)
        return mse_dict#, mape_dict


def postprocess_raw_metrics(acc_dict, f1_dict):
    acc_overall_avg, f1_overall_avg = compute_dict_average(acc_dict), \
                                      compute_dict_average(f1_dict)
    acc_category_avgs_dict, f1_category_avgs_dict = compute_category_avgs(acc_dict), \
                                                    compute_category_avgs(f1_dict)
    acc_avg_across_categories, f1_avg_across_categories = compute_dict_average(acc_category_avgs_dict), \
                                                          compute_dict_average(f1_category_avgs_dict)
    acc_dict.update(acc_category_avgs_dict)
    f1_dict.update(f1_category_avgs_dict)

    acc_dict["overall_avg"], f1_dict["overall_avg"] = acc_overall_avg, f1_overall_avg
    acc_dict["across_categories_avg"], f1_dict["across_categories_avg"] = [acc_avg_across_categories,
                                                                           f1_avg_across_categories]

    acc_dict = append_suffix(acc_dict, "_acc")
    f1_dict = append_suffix(f1_dict, "_f1")

    return acc_dict, f1_dict


def compute_category_avgs(metric_dict):
    category_dict = {}
    for category_name, category_keys in summary_key_dict.items():
        category_values = [v for k, v in metric_dict.items() if k in category_keys]
        if len(category_values) < 1:
            continue
        category_mean = np.mean(category_values)
        category_dict[category_name + "_avg"] = category_mean
    return category_dict
