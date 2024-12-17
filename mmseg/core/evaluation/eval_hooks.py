import os.path as osp

from mmcv.utils import print_log
from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=False, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs
        self.save_thres=eval_kwargs.get('save_thres',10.0)
        self.save_path=eval_kwargs.get('save_path')
        self.save_metric=eval_kwargs.get('save_metric','mIoU')
        self.metric_max=0.0

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Call evaluate function of dataset."""
        runner.save_checkpoint(self.save_path,
                               filename_tmpl='latest.pth')
        print_log(f'successful save checkpoint , save path : {self.save_path}/latest.pth', runner.logger)

        eval_res = self.dataloader.dataset.cancer_evaluate(
            results, logger=runner.logger,runner=runner, **self.eval_kwargs)

        # max_matric=0.0
        # for eval in eval_res.keys():
        #     max_matric+=eval_res[eval][self.save_metric]
        # max_matric=max_matric/len(eval_res.keys())
        if self.metric_max<=eval_res['final'][self.save_metric]:
            save_name=f'{self.save_metric}_{eval_res["final"][self.save_metric]}.pth'
            runner.save_checkpoint(self.save_path,
                                   filename_tmpl=save_name)
            print_log(f'successful save checkpoint , save path : {self.save_path}/{save_name}', runner.logger)

            self.metric_max = eval_res['final'][self.save_metric]



class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
