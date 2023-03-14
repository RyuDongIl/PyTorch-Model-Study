# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 19:37:54 2022

@author: Dongil Ryu
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size=2):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_size, dropout_ratio):
        super().__init__()

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding)을 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # GRU Layer
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(embed_dim, hidden_size, bidirectional=True)
        
        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    # 인코더는 소스 문장을 입력으로 받아 문맥 벡터(context vector)를 반환        
    def forward(self, src):
        # src: [단어 개수, 배치 크기]: 각 단어의 인덱스(index) 정보
        embedded = self.dropout(self.embedding(src))
        # embedded: [단어 개수, 배치 크기, 임베딩 차원]

        outputs, h_n = self.rnn(embedded)
        # outputs: [단어 개수, 배치 크기, 2*히든 차원]: 현재 단어의 출력 정보
        # h_n: [2*레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보

        return outputs


class Alignment(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.attn = nn.Linear(hidden_dim*3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        
    def forward(self, trg_hidden, encoder_outputs):
        # trg_hidden: [배치 크기, 히든 차원]
        # encoder_outputs: [단어 개수, 배치 크기, 2*히든 차원]
        
        src_len = encoder_outputs.shape[0]
        
        trg_hidden = trg_hidden.repeat(src_len, 1, 1)
        # trg_hidden: [단어 개수, 배치 크기, 히든 차원]
        
        e = torch.tanh(self.attn(torch.cat((trg_hidden, encoder_outputs), dim=-1))) # e: [단어 개수, 배치 크기, 히든 차원]
        e = self.v(e).squeeze(2) # e: [단어 개수, 배치 크기]
        e = e.transpose(0, 1) # e: [배치 크기, 단어 개수]
        
        return F.softmax(e, dim=1)
        
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_size, deep_dim, dropout_ratio):
        super().__init__()

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding) 말고 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(output_dim, embed_dim)

        # GRU Layer
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(embed_dim + 2*hidden_size, hidden_size)
        
        # Alignment
        self.align = Alignment(hidden_size)
        
        # for first hidden
        self.W_s = nn.Linear(hidden_size, hidden_size)
        
        # making output
        self.maxout = Maxout(embed_dim + 3*hidden_size, deep_dim, pool_size=2)
        self.output_dim = output_dim
        self.fc_out = nn.Linear(deep_dim, output_dim)
        
        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    # 디코더는 현재까지 출력된 문장에 대한 정보를 입력으로 받아 타겟 문장을 반환     
    def forward(self, encoder_outputs, input, hidden=None):
        # encoder_outputs: [단어 개수, 배치 크기, 2*히든 차원]
        # input: [배치 크기]: 단어의 개수는 항상 1개이도록 구현
        # hidden: [레이어 수 = 1, 배치 크기, 히든 차원]
        if hidden is None:
            hidden = torch.tanh(self.W_s(encoder_outputs[0, :, self.hidden_size:]))
            hidden = hidden.unsqueeze(0)

        input = input.unsqueeze(0)
        # input: [타겟 단어수 = 1, 배치 크기]
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [1, 배치 크기, 임베딩 차원]
        
        a = self.align(hidden, encoder_outputs).unsqueeze(1)
        # a: [배치 크기, 1, 단어 개수]
        context = torch.bmm(a, encoder_outputs.transpose(0, 1)).transpose(0, 1)
        # context: [1, 배치 크기, 2*히든 차원]
        new_input = torch.cat((embedded, context), dim=-1)
        # new_input: [1, 배치 크기, 임베딩 차원 + 2*히든 차원]
        
        output, next_hidden = self.rnn(new_input, hidden)
        # output: [단어 개수 = 1, 배치 크기, 히든 차원]
        # next_hidden: [레이어 개수 = 1, 배치 크기, 히든 차원]

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        context = context.squeeze(0)

        prediction = self.dropout(self.maxout(torch.cat((embedded, output, context), dim=1)))
        prediction = self.dropout(self.fc_out(prediction))
        # prediction: [배치 크기, 출력 차원]

        return prediction, next_hidden
    

class Attention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif name.startswith('decoder.align.v.'):
                nn.init.constant_(param, 0)
            elif name.startswith('decoder.align.attn.'):
                nn.init.normal_(param, mean=0, std=0.001)
            elif 'rnn.weight' in name:
                nn.init.orthogonal_(param)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.01)

    # 학습할 때는 완전한 형태의 소스 문장, 타겟 문장, teacher_forcing_ratio를 넣기
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [단어 개수, 배치 크기]
        # trg: [단어 개수, 배치 크기]
        # 먼저 인코더를 거쳐 문맥 벡터(context vector)를 추출
        encoder_outputs = self.encoder(src)

        # 디코더(decoder)의 최종 결과를 담을 텐서 객체 만들기
        trg_len = trg.shape[0] # 단어 개수
        batch_size = trg.shape[1] # 배치 크기
        trg_vocab_size = self.decoder.output_dim # 출력 차원
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 첫 번째 입력은 항상 <sos> 토큰
        input = trg[0, :]
        hidden = None

        # 타겟 단어의 개수만큼 반복하여 디코더에 포워딩(forwarding)
        for t in range(1, trg_len):
            output, hidden = self.decoder(encoder_outputs, input, hidden)

            outputs[t] = output # FC를 거쳐서 나온 현재의 출력 단어 정보
            top1 = output.argmax(1) # 가장 확률이 높은 단어의 인덱스 추출

            # teacher_forcing_ratio: 학습할 때 실제 목표 출력(ground-truth)을 사용하는 비율
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1 # 현재의 출력 결과를 다음 입력에서 넣기
        
        return outputs