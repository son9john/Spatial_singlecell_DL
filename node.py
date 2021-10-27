import logging

import torch

import hideandseek as hs

# %%
log = logging.getLogger(__name__)

# %%
class AENode(hs.N.Node):
    def update(self, data, device, prefix=''):
        try:
            x = data.to(device)
            N = len(x)

            if self.amp:
                # Mixed precision for acceleration
                with torch.cuda.amp.autocast():
                    x_hat = self.model(x)
                    loss = self.criterion(x_hat, x)
            else:
                x_hat = self.model(x)
                loss = self.criterion(x_hat, x)

            self.op.zero_grad()
            loss.backward()
            self.op.step()
            self.train_meter.step(loss.item(), N)

            self.print(f'{prefix}[Loss: {loss.item():.7f} (Avg: {self.train_meter.avg:.7f})]')
            return loss.item()
        except Exception as e:
            log.warning(e)
            import pdb; pdb.set_trace()
