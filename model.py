import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Policy networks. The neural policy networks were trained by minimizing the negative
log-likelihood of selecting the transformation a that was used in the literature
to make molecule m. This is essentially supervised multi-class classification. To
evaluate the accuracy, reactions that are not covered in the rule set are excluded.
Training was carried out using stochastic gradient descent (ADAM optimizer64)
within 1–2 days on a single NVIDIA K80 graphics processing unit. The Keras
neural network framework was employed, using Theano as the backend65,66.

Expansion policy network. Molecules are represented by real vectors in the form
of counted extended-connectivity fingerprints (ECFP4)67, which are first modulofolded
to 1,000,000 dimensions, and then ln(x+1)-preprocessed. After that,
a variance threshold is applied to remove rare features, leaving 32,681 dimensions.
For the machine-learning model, we used a 1+5-layer highway network with
exponential linear unit (ELU) nonlinearities37,38. A dropout ratio of 0.3 was applied
after the first affine transformation to 512 dimensions, and a dropout ratio of
0.1 was applied after each of the five highway layers. The last layer of the neural
networks is a softmax, which outputs a probability distribution over all actions
(transformations) p(a|m), which forms the policy (see Extended Data Fig. 4).
'''

class Highway(nn.Module):
    '''
    Adapted from https://github.com/kefirski/pytorch_Highway
    '''
    def __init__(self, size, num_layers, f, dropout,
                head=False, input_size=None):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.f = f
        self.dropout = nn.Dropout(dropout)
        if head:
            assert input_size is not None
            self.nonlinear = nn.ModuleList([nn.Linear(input_size, size)])
            self.linear = nn.ModuleList([nn.Linear(input_size, size)])
            self.gate = nn.ModuleList([nn.Linear(input_size, size)])
        else:
            self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
            self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
            self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
            
    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear
            x = self.dropout(x) # TODO: check if this is correct
        return x

# in the Nature paper, ELU non-linearity was used
class TemplateNN_Highway(nn.Module):
    def __init__(self, output_size, size=512, num_layers_body=5, 
                dropout_head=0.3, dropout_body=0.1,
                f=F.elu, input_size=32681):
        super(TemplateNN_Highway, self).__init__()
        self.highway_head = Highway(
                size=size, num_layers=1, f=f, dropout=dropout_head, 
                head=True, input_size=input_size
            )
        if num_layers_body <= 0:
            self.highway_body = None
        else:
            self.highway_body = Highway(
                    size=size, num_layers=num_layers_body, 
                    f=f, dropout=dropout_body
                )
        
        self.classifier = nn.Linear(size, output_size)

    def forward(self, fp):
        if self.highway_body:
            embedding = self.highway_body(self.highway_head(fp))
        else:
            embedding = self.highway_head(fp)
        return self.classifier(embedding).squeeze(dim=1)

class TemplateNN_FC(nn.Module):
    ''' Following Segler's 2017 paper, we also try the one-layer (512) ELU FC network, as it may give better results on USPTO-50K
    '''
    def __init__(self, output_size, size=512,
                dropout=0.1, input_size=32681):
        super(TemplateNN_FC, self).__init__()
        self.fc = nn.Sequential(*
                    [nn.Linear(input_size, size),
                    nn.ELU(),
                    nn.Dropout(dropout)]
                )
        self.classifier = nn.Linear(size, output_size)

    def forward(self, fp):
        embedding = self.fc(fp)
        return self.classifier(embedding).squeeze(dim=1)