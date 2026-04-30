import torch

class GradSyncer:
    def __init__(self, module: torch.nn.Module, enabled: bool = True):
        self._module = module
        self.enabled = enabled

        self.registered_params = {}
        self.waiting_for_sync = []

        # Trick below from torch/distributed/optim/apply_optimizer_in_backward.py. It creates a new view of the variable,
        # but because the grads should be available also in this view, we can register a hook on the view and get the
        # already computed grads from there.

        def register_param_syncer(param):
            if id(param) in self.registered_params:
                return

            self.registered_params[id(param)] = param.view_as(param).grad_fn.next_functions[0][0]

            def hook(*_):
                if self.enabled and self._module.training and param.grad is not None:
                    self.waiting_for_sync.append(torch.distributed.all_reduce(param.grad.contiguous(), async_op=True))

            self.registered_params[id(param)].register_hook(hook)

        for param in module.parameters():
            register_param_syncer(param)

        print(f"Registered {len(self.registered_params)} parameters for syncing.")

    def sync(self):
        for a in self.waiting_for_sync:
            a.wait()

        self.waiting_for_sync.clear()
