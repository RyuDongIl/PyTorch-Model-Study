import torch.nn as nn

LAYERS_INFO = {
    'A': [[1, 1, 2, 2, 2], []],
    'B': [[2, 2, 2, 2, 2], []],
    'C': [[2, 2, 3, 3, 3], ['3_3', '4_3', '5_3']],
    'D': [[2, 2, 3, 3, 3], []],
    'E': [[2, 2, 4, 4, 4], []]
    }

def conv_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU()
    )

def conv_layer(block_info):
    block_list = [conv_block(in_f, out_f, kernel_size=k, padding=p)
                  for in_f, out_f, k, p in block_info]
    block_list.append(nn.MaxPool2d(2, stride=2))
    return nn.Sequential(*block_list)

class VGGNetTest(nn.Module):
    
    def __init__(self, config='A', num_classes=1000, img_size=224):
        super(VGGNetTest, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.layer_info = LAYERS_INFO[self.config]
        
        block_infos = self.get_block_infos()
        self.encoder = nn.Sequential(*[conv_layer(block_info) for block_info in block_infos])

        self.decoder = nn.Sequential(
                            nn.Conv2d(512, 4096, kernel_size=(img_size // 32), padding=0),
                            nn.ReLU(),
                            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
                            nn.ReLU(),
                            nn.Conv2d(4096, self.num_classes, kernel_size=1, padding=0))
    
    def get_block_infos(self):
        enc_sizes = [3, 64, 128, 256, 512, 512]
        
        block_infos = list()
        for l_idx, cnt in enumerate(self.layer_info[0]):
            block_info = list()
            for c_idx in range(cnt):
                in_f, out_f = l_idx + int(bool(c_idx)), l_idx + 1
                if f'{l_idx+1}_{c_idx+1}' in self.layer_info[1]:
                    block_info.append((enc_sizes[in_f], enc_sizes[out_f], 1, 0))
                else:
                    block_info.append((enc_sizes[in_f], enc_sizes[out_f], 3, 1))
            block_infos.append(block_info)
        
        return block_infos

    def forward(self, inputs):
        out = self.encoder(inputs)
        out = self.decoder(out)
        return out