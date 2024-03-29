# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 19:13:06 2022

@author: Dongil Ryu
"""
import random
import torch
import torch.nn as nn


# 인코더(Encoder) 아키텍처 정의
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers, dropout_ratio):
        super().__init__()

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding)을 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # LSTM 레이어
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout_ratio)
        
        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    # 인코더는 소스 문장을 입력으로 받아 문맥 벡터(context vector)를 반환        
    def forward(self, src):
        # src: [단어 개수, 배치 크기]: 각 단어의 인덱스(index) 정보
        embedded = self.dropout(self.embedding(src))
        # embedded: [단어 개수, 배치 크기, 임베딩 차원]

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [단어 개수, 배치 크기, 히든 차원]: 현재 단어의 출력 정보
        # hidden: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보
        # cell: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보

        # 문맥 벡터(context vector) 반환
        return hidden, cell


# 디코더(Decoder) 아키텍처 정의
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout_ratio):
        super().__init__()

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding) 말고 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(output_dim, embed_dim)

        # LSTM 레이어
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout_ratio)
        
        # FC 레이어 (인코더와 구조적으로 다른 부분)
        self.output_dim = output_dim
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    # 디코더는 현재까지 출력된 문장에 대한 정보를 입력으로 받아 타겟 문장을 반환     
    def forward(self, input, hidden, cell):
        # input: [배치 크기]: 단어의 개수는 항상 1개이도록 구현
        # hidden: [레이어 개수, 배치 크기, 히든 차원]
        # cell = context: [레이어 개수, 배치 크기, 히든 차원]
        input = input.unsqueeze(0)
        # input: [단어 개수 = 1, 배치 크기]
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [단어 개수, 배치 크기, 임베딩 차원]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [단어 개수 = 1, 배치 크기, 히든 차원]: 현재 단어의 출력 정보
        # hidden: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보
        # cell: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보

        # 단어 개수는 어차피 1개이므로 차원 제거
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [배치 크기, 출력 차원]
        
        # (현재 출력 단어, 현재까지의 모든 단어의 정보, 현재까지의 모든 단어의 정보)
        return prediction, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    # 학습할 때는 완전한 형태의 소스 문장, 타겟 문장, teacher_forcing_ratio를 넣기
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [단어 개수, 배치 크기]
        # trg: [단어 개수, 배치 크기]
        # 먼저 인코더를 거쳐 문맥 벡터(context vector)를 추출
        hidden, cell = self.encoder(src)

        # 디코더(decoder)의 최종 결과를 담을 텐서 객체 만들기
        trg_len = trg.shape[0] # 단어 개수
        batch_size = trg.shape[1] # 배치 크기
        trg_vocab_size = self.decoder.output_dim # 출력 차원
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 첫 번째 입력은 항상 <sos> 토큰
        input = trg[0, :]

        # 타겟 단어의 개수만큼 반복하여 디코더에 포워딩(forwarding)
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = output # FC를 거쳐서 나온 현재의 출력 단어 정보
            top1 = output.argmax(1) # 가장 확률이 높은 단어의 인덱스 추출

            # teacher_forcing_ratio: 학습할 때 실제 목표 출력(ground-truth)을 사용하는 비율
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1 # 현재의 출력 결과를 다음 입력에서 넣기
        
        return outputs
    