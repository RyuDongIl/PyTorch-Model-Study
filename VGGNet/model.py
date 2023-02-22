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

class VGGNet(nn.Module):
    
    def __init__(self, config='A', num_classes=1000, img_size=224):
        super(VGGNet, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.layer_info = LAYERS_INFO[self.config]
        
        block_infos = self.get_block_infos()
        self.encoder = nn.Sequential(*[conv_layer(block_info) for block_info in block_infos])

        self.decoder = nn.Sequential(
                            nn.Linear((img_size // 32) * (img_size // 32) * 512, 4096),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(4096, 4096),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(4096, self.num_classes))

        self.initialize_weights()
    
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
    
    def initialize_weights(self):
        for m in self.modules():
            # convolution kernel의 weight를 He initialization을 적용한다.
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                
                # bias는 상수 0으로 초기화 한다.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)              
    
    def forward(self, inputs):
        out = self.encoder(inputs)
        out = out.view(out.size(0), -1)
        out = self.decoder(out)
        return out