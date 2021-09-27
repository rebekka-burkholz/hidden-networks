import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from args import args as parser_args


DenseConv = nn.Conv2d


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, scoresBias, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        outBias = scoresBias.clone()
        #print(out.size())
        #print(outBias.size())
        #threshold = cat([scores.flatten(), scoresBias.flatten()])
        val, _ = torch.cat([scores.flatten(), scoresBias.flatten()]).sort()
        j = int((1 - k) * val.numel())
        threshold = val[j]
        # flat_out and out access the same memory.
        #flat_out = out.flatten()
        #flat_out[idx[:j]] = 0
        #flat_out[idx[j:]] = 1
        out = torch.where(out <= threshold, 0.0, 1.0)
        outBias = torch.where(outBias <= threshold, 0.0, 1.0)
#        print(out.size())
#        print(outBias.size())
        return out, outBias

    @staticmethod
    def backward(ctx, g, gbias):
        #print("send back")
        # send the gradient g straight-through on the backward pass.
        return g, gbias, None


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        #print(self.bias.size())
        self.scoresBias = nn.Parameter(torch.Tensor(self.bias.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        #nn.init.kaiming_uniform_(self.scoresBias, a=math.sqrt(5))
        fan = nn.init._calculate_correct_fan(self.scores, 'fan_in')
        bound = math.sqrt(6.0/fan)
        nn.init.uniform_(self.scoresBias, -bound, bound)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()
    
    @property
    def clamped_scoresBias(self):
        return self.scoresBias.abs()

    def forward(self, x):
        subnet, subnetBias = GetSubnet.apply(self.clamped_scores, self.clamped_scoresBias, self.prune_rate)
        #subnet, subnetBias = GetSubnet.apply(self.scores.abs(), self.scoresBias.abs(), self.prune_rate)
        w = self.weight * subnet
        b = self.bias * subnetBias
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x


"""
Sample Based Sparsification
"""


class StraightThroughBinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores, scoresBias):
        output = (torch.rand_like(scores) < scores).float()
        outputBias = (torch.rand_like(scoresBias) < scoresBias).float()
        return output, outputBias

    @staticmethod
    def backward(ctx, grad_outputs, grad_outputs_bias):
        return grad_outputs, grad_outputs_bias


class BinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores, scoresBias):
        output = (torch.rand_like(scores) < scores).float()
        outputBias = (torch.rand_like(scoresBias) < scoresBias).float()
        ctx.save_for_backward(output)
        ctx.save_for_backward(outputBias)

        return output, outputBias

    @staticmethod
    def backward(ctx, grad_outputs, grad_outputs_bias):
        subnet, subnetBias, = ctx.saved_variables

        grad_inputs = grad_outputs.clone()
        grad_inputs_bias = grad_outputs_bias.clone()
        grad_inputs[subnet == 0.0] = 0.0
        grad_inputs_bias[subnetBias == 0.0] = 0.0
        return grad_inputs, grad_inputs_bias


# Not learning weights, finding subnet
class SampleSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.scoresBias = nn.Parameter(torch.Tensor(self.bias.size()))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                torch.ones_like(self.scores) * parser_args.score_init_constant
            )
            self.scoresBias.data = (
                torch.ones_like(self.scoresBias) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.scoresBias, a=math.sqrt(5))
            
    @property
    def clamped_scores(self):
        return torch.sigmoid(self.scores)
    
    @property
    def clamped_scoresBias(self):
        return torch.sigmoid(self.scoresBias)

    def forward(self, x):
        subnet = StraightThroughBinomialSample.apply(self.clamped_scores)
        subnetBias = StraightThroughBinomialSample.apply(self.clamped_scoresBias)
        w = self.weight * subnet
        b = self.bias * subnetBias
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )

        return x


"""
Fixed subnets 
"""


class FixedSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.scoresBias = nn.Parameter(torch.Tensor(self.bias.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        #nn.init.kaiming_uniform_(self.scoresBias, a=math.sqrt(5))
        fan = nn.init._calculate_correct_fan(self.scores, 'fan_in')
        bound = math.sqrt(6.0/fan)
        nn.init.uniform_(self.scoresBias, -bound, bound)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print("prune_rate_{}".format(self.prune_rate))

    def set_subnet(self):
        output = self.clamped_scores().clone()
        outputBias = self.clamped_scoresBias().clone()
        #_, idx = self.clamped_scores().flatten().abs().sort()
        #p = int(self.prune_rate * self.clamped_scores().numel())
        #flat_oup = output.flatten()
        #flat_oup[idx[:p]] = 0
        #flat_oup[idx[p:]] = 1
        val, _ = torch.cat([self.clamped_scores().flatten().abs(), self.clamped_scoresBias().flatten().abs()]).sort()
        j = int((1-self.prune_rate) * val.numel())
        threshold = val[j]
        out = torch.where(out <= threshold, 0, 1)
        outBias = torch.where(outBias <= threshold, 0, 1)
        self.scores = torch.nn.Parameter(output)
        self.scores.requires_grad = False
        self.scoresBias = torch.nn.Parameter(outputBias)
        self.scoresBias.requires_grad = False
        
        
    def clamped_scores(self):
        return self.scores.abs()

    def clamped_scoresBias(self):
        return self.scoresBias.abs()

    def get_subnet(self):
        return self.weight * self.scores, self.bias * self.scoresBias

    def forward(self, x):
        w, b = self.get_subnet()
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x

